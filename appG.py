import base64
import hashlib
import html
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import boto3
import streamlit as st
from botocore.exceptions import ClientError
from streamlit_mic_recorder import speech_to_text
from supabase import create_client, Client


# ==========================================
# PAGE SETUP
# ==========================================
st.set_page_config(page_title="Kindred", page_icon="🕊️", layout="centered")
st.markdown(
    """
    <style>
    .stApp,
    [data-testid="stAppViewContainer"] {
        background-color: #465064;
    }
    .stApp {
        color: #f4f0f7;
    }
    [data-testid="stHeader"] {
        background-color: rgba(70, 80, 100, 0.92);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# SUPABASE CLIENT
# ==========================================
@st.cache_resource
def get_supabase() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_ANON_KEY"]
    return create_client(url, key)

supabase = get_supabase()


# ==========================================
# AMAZON BEDROCK CONFIGURATION
# ==========================================
MODELS = [
    "eu.amazon.nova-micro-v1:0",
    "eu.amazon.nova-lite-v1:0",
]

try:
    aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
    aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
    aws_region = st.secrets.get("AWS_REGION", "eu-west-1")
except Exception:
    st.error(
        "AWS credentials are missing. Add AWS_ACCESS_KEY_ID, "
        "AWS_SECRET_ACCESS_KEY, and AWS_REGION to your Streamlit secrets."
    )
    st.stop()

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=aws_region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
)


# ==========================================
# SPIRIT CONFIGURATION
# ==========================================
SPIRITS = {
    "fireice": {
        "name": "Glowing Fire&Ice",
        "file": "spirits/FireIce_pingpong.mp4",
        "fallback": "spirits/mist.png",
        "description": "Warm, wrapping, comforting",
        "animation": "fireice",
    },
    "water": {
        "name": "Living Water",
        "file": "spirits/water.mp4",
        "fallback": "spirits/water.png",
        "description": "Reflective, flowing, cleansing",
        "animation": "water",
    },
    "starlight": {
        "name": "Star Fragment",
        "file": "spirits/starlight.mp4",
        "fallback": "spirits/starlight.png",
        "description": "Hopeful, bright, wonder-filled",
        "animation": "starlight",
    },
    "growth": {
        "name": "Gentle Leaf",
        "file": "spirits/growth.mp4",
        "fallback": "spirits/growth.png",
        "description": "Grounded, unfolding, strengthening",
        "animation": "growth",
    },
}

# Questionnaire → spirit mapping
ESSENCE_MAP = {
    "☁️ Fire&Ice": "fireice",
    "💧 Water": "water",
    "⭐ Starlight": "starlight",
    "🌿 Growth": "growth",
}

SPIRIT_KEY_ALIASES = {
    "mist": "fireice",
}


# ==========================================
# SYSTEM PROMPTS
# ==========================================
MAIN_PROMPT = """
You are Kindred — a gentle, empathetic listening presence.
Your personality: warm, patient, non-judgmental, thoughtful, soft-spoken, and neutral.
You are a safe space: be respectful, inclusive, and understanding.

Your purpose is to hold space for whatever the user is feeling.
You do not need to fix everything — just be there.

Guidelines:
- Speak in a warm, gentle, conversational tone.
- Keep responses short and conversational — 2 to 4 sentences max.
- Validate feelings before offering perspective.
- Be inclusive; never assume or reinforce stereotypes.
- Never be preachy or overly cheerful.
- If you don't know something, be honest.
- If someone may be in immediate danger, encourage them to contact local emergency services or a trusted person.
- For South African users in crisis: SADAG 0800 567 567, Lifeline 0861 322 322.
"""

KIDS_PROMPT = """
You are Kindred Kids — a friendly, patient learning companion for children up to age 12.

Rules:
- Speak simply, clearly, and warmly, like a kind teacher.
- Guide children through problems rather than simply giving final answers.
- Use examples, simple steps, and encouragement.
- If they are stuck, say: "Let's try one step at a time together"
- For sensitive, risky, or personal matters, encourage them to speak with a trusted grown-up.
- Be encouraging; celebrate effort, not just being right.
- Keep responses short and age-appropriate.
- Never ask for a child's name, school, address, or other personal details.
- Do not claim to be a human, therapist, or emergency service.
"""


# ==========================================
# SESSION STATE
# ==========================================
defaults = {
    "messages": [],
    "mode": "main",
    "spirit_generated": False,
    "spirit_text": "",
    "voice_speed": 1.0,
    "auto_speak_text": "",
    "user": None,
    "is_premium": False,
    "auth_view": "login",
    "auth_error": "",
    "auth_success": "",
    "spirit_key": None,        # which spirit the user has (Fire&Ice/water/starlight/growth)
    "spirit_speaking": False,  # true while Kindred is responding
    "show_questionnaire": False,
    "guest_usage_id": None,
    "usage_date": "",
    "usage_count": 0,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ==========================================
# FREE DAILY MESSAGE CAP
# ==========================================
# Premium is unlimited. Guests and free accounts share a daily Bedrock budget
# (chat + whispers). Override with Streamlit secret FREE_DAILY_MESSAGE_LIMIT.
SAST = timezone(timedelta(hours=2))
try:
    FREE_DAILY_MESSAGE_LIMIT = int(st.secrets.get("FREE_DAILY_MESSAGE_LIMIT", 30))
except (TypeError, ValueError):
    FREE_DAILY_MESSAGE_LIMIT = 30
if FREE_DAILY_MESSAGE_LIMIT < 1:
    FREE_DAILY_MESSAGE_LIMIT = 30

CAP_REACHED_MESSAGE = (
    "You've used today's free messages — that's okay. Kindred will be here "
    "again tomorrow. Premium removes the daily cap if you ever want more space."
)


def _usage_today() -> str:
    return datetime.now(SAST).date().isoformat()


@st.cache_resource
def _usage_memory_store() -> dict:
    """Process-level fallback so a refresh cannot reset the cap on this server."""
    return {}


def _client_ip() -> Optional[str]:
    try:
        ip = getattr(st.context, "ip_address", None)
        if ip:
            return str(ip)
        headers = getattr(st.context, "headers", None)
        if not headers:
            return None
        forwarded = headers.get("X-Forwarded-For") or headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = headers.get("X-Real-IP") or headers.get("x-real-ip")
        if real_ip:
            return str(real_ip)
    except Exception:
        return None
    return None


def _usage_fingerprint() -> str:
    user = st.session_state.user
    if user and user != "guest":
        return f"user:{user.id}"

    salt = str(st.secrets.get("USAGE_FINGERPRINT_SALT", "kindred"))
    ip = _client_ip()
    if ip:
        digest = hashlib.sha256(f"{salt}|ip|{ip}".encode()).hexdigest()[:24]
        return f"guest:{digest}"

    if not st.session_state.guest_usage_id:
        existing = None
        try:
            existing = st.query_params.get("kid")
            if isinstance(existing, list):
                existing = existing[0] if existing else None
        except Exception:
            existing = None
        if existing:
            st.session_state.guest_usage_id = str(existing)
        else:
            st.session_state.guest_usage_id = hashlib.sha256(
                os.urandom(16)
            ).hexdigest()[:24]
            try:
                st.query_params["kid"] = st.session_state.guest_usage_id
            except Exception:
                pass
    digest = hashlib.sha256(
        f"{salt}|sid|{st.session_state.guest_usage_id}".encode()
    ).hexdigest()[:24]
    return f"guest:{digest}"


def _parse_rpc_payload(data):
    if data is None:
        return None
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return None
    if isinstance(data, list) and data:
        data = data[0]
    return data if isinstance(data, dict) else None


def _memory_usage_count(fingerprint: str) -> int:
    store = _usage_memory_store()
    return int(store.get(f"{fingerprint}:{_usage_today()}", 0))


def _memory_consume(fingerprint: str, limit: int) -> tuple:
    store = _usage_memory_store()
    key = f"{fingerprint}:{_usage_today()}"
    count = int(store.get(key, 0))
    if count >= limit:
        return False, count
    count += 1
    store[key] = count
    return True, count


def get_daily_usage_count() -> int:
    if st.session_state.is_premium:
        return 0
    fingerprint = _usage_fingerprint()
    today = _usage_today()
    remote = None
    try:
        result = supabase.rpc(
            "get_daily_message_count", {"p_fingerprint": fingerprint}
        ).execute()
        if result.data is not None:
            remote = int(result.data)
    except Exception:
        remote = None

    memory = _memory_usage_count(fingerprint)
    if st.session_state.get("usage_date") != today:
        st.session_state.usage_date = today
        st.session_state.usage_count = 0
    combined = max(int(st.session_state.usage_count or 0), memory)
    if remote is not None:
        combined = max(combined, remote)
        st.session_state.usage_count = combined
        st.session_state.usage_date = today
    return combined


def remaining_free_messages() -> Optional[int]:
    if st.session_state.is_premium:
        return None
    return max(0, FREE_DAILY_MESSAGE_LIMIT - get_daily_usage_count())


def consume_free_message() -> bool:
    """Reserve one free Bedrock call. Premium always allowed."""
    if st.session_state.is_premium:
        return True

    fingerprint = _usage_fingerprint()
    limit = FREE_DAILY_MESSAGE_LIMIT
    today = _usage_today()

    try:
        result = supabase.rpc(
            "consume_daily_message",
            {"p_fingerprint": fingerprint, "p_limit": limit},
        ).execute()
        payload = _parse_rpc_payload(result.data)
        if payload is not None and "allowed" in payload:
            count = int(payload.get("count") or 0)
            allowed = bool(payload.get("allowed"))
            st.session_state.usage_date = today
            st.session_state.usage_count = count
            _usage_memory_store()[f"{fingerprint}:{today}"] = count
            return allowed
    except Exception:
        pass

    allowed, count = _memory_consume(fingerprint, limit)
    st.session_state.usage_date = today
    st.session_state.usage_count = count
    return allowed


def render_free_cap_notice() -> None:
    st.info(CAP_REACHED_MESSAGE)


# ==========================================
# AUTH HELPERS
# ==========================================
def get_profile(user_id: str) -> dict:
    try:
        result = supabase.table("profiles").select("*").eq("id", user_id).single().execute()
        return result.data or {}
    except Exception:
        return {}


def login(email: str, password: str):
    try:
        result = supabase.auth.sign_in_with_password({"email": email, "password": password})
        st.session_state.user = result.user
        profile = get_profile(result.user.id)
        st.session_state.is_premium = profile.get("is_premium", False)
        st.session_state.spirit_key = profile.get("spirit_key", None)
        st.session_state.usage_date = ""
        st.session_state.usage_count = 0
        st.session_state.auth_error = ""
        st.rerun()
    except Exception:
        st.session_state.auth_error = "Incorrect email or password. Please try again."


def signup(email: str, password: str):
    try:
        result = supabase.auth.sign_up({"email": email, "password": password})
        if result.user:
            st.session_state.auth_success = (
                "Account created! Check your email to confirm, then log in."
            )
            st.session_state.auth_view = "login"
            st.session_state.auth_error = ""
        else:
            st.session_state.auth_error = "Something went wrong. Please try again."
    except Exception as e:
        err = str(e).lower()
        if "already" in err or "exists" in err:
            st.session_state.auth_error = "An account with that email already exists."
        elif "password" in err:
            st.session_state.auth_error = "Password must be at least 6 characters."
        else:
            st.session_state.auth_error = "Could not create account. Please try again."


def logout():
    supabase.auth.sign_out()
    for key in ["user", "is_premium", "messages", "spirit_generated", "spirit_text",
                "auto_speak_text", "spirit_key", "spirit_speaking", "show_questionnaire",
                "guest_usage_id", "usage_date", "usage_count"]:
        st.session_state[key] = defaults[key]
    st.rerun()


def save_spirit(spirit_key: str):
    """Save the user's chosen spirit to their profile."""
    if st.session_state.user and st.session_state.user != "guest":
        try:
            supabase.table("profiles").update(
                {"spirit_key": spirit_key}
            ).eq("id", st.session_state.user.id).execute()
            st.session_state.spirit_key = spirit_key
        except Exception:
            pass


# ==========================================
# AUTH UI
# ==========================================
def render_auth():
    st.title("🕊️ Kindred")
    st.markdown("##### A gentle space to say what's on your mind.")
    st.divider()

    st.markdown("**Continue without an account**")
    st.caption("Free access — no sign-up needed. A usage ID helps enforce the daily limit.")
    if st.button("Enter as Guest", use_container_width=True):
        st.session_state.user = "guest"
        st.session_state.is_premium = False
        st.session_state.usage_date = ""
        st.session_state.usage_count = 0
        st.rerun()

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Log In", use_container_width=True,
                     type="primary" if st.session_state.auth_view == "login" else "secondary"):
            st.session_state.auth_view = "login"
            st.session_state.auth_error = ""
            st.session_state.auth_success = ""
            st.rerun()
    with col2:
        if st.button("Create Account", use_container_width=True,
                     type="primary" if st.session_state.auth_view == "signup" else "secondary"):
            st.session_state.auth_view = "signup"
            st.session_state.auth_error = ""
            st.session_state.auth_success = ""
            st.rerun()

    st.markdown("")

    if st.session_state.auth_success:
        st.success(st.session_state.auth_success)

    if st.session_state.auth_view == "login":
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In", use_container_width=True, type="primary")
            if submitted:
                if email and password:
                    login(email, password)
                else:
                    st.session_state.auth_error = "Please enter your email and password."
        if st.session_state.auth_error:
            st.error(st.session_state.auth_error)
    else:
        with st.form("signup_form"):
            email = st.text_input("Email")
            password = st.text_input("Password (min. 6 characters)", type="password")
            submitted = st.form_submit_button("Create Account", use_container_width=True, type="primary")
            if submitted:
                if email and password:
                    signup(email, password)
                    st.rerun()
                else:
                    st.session_state.auth_error = "Please fill in all fields."
        if st.session_state.auth_error:
            st.error(st.session_state.auth_error)

    st.divider()
    st.caption(
        "Premium features (voice, conversation history, Kindred Spirit, Kids Mode) "
        "require an account. Free chat is always available — no account needed."
    )


# ==========================================
# SHOW AUTH SCREEN IF NOT LOGGED IN
# ==========================================
if not st.session_state.user:
    render_auth()
    st.stop()


# ==========================================
# SPIRIT HELPERS
# ==========================================

def render_spirit(spirit_key: str, speaking: bool = False):
    """Render the animated spirit component using video."""
    spirit_key = SPIRIT_KEY_ALIASES.get(spirit_key, spirit_key)
    spirit = SPIRITS.get(spirit_key)
    if spirit is None:
        st.warning("Your saved spirit isn't available. Please update your spirit to continue.")
        return

    video_path = spirit["file"]
    fallback_path = spirit["fallback"]
    animation = spirit["animation"]

    # Use video if available, fall back to image
    use_video = os.path.exists(video_path)
    media_path = video_path if use_video else fallback_path

    if not os.path.exists(media_path):
        return

    # Encode media to base64
    with open(media_path, "rb") as f:
        media_b64 = base64.b64encode(f.read()).decode()

    # Per-essence glow and particle colours
    essence_styles = {
        "fireice":   {"glow": "rgba(255, 210, 120, 0.5)",  "particle_color": "#ffddaa"},
        "water":     {"glow": "rgba(80,  160, 255, 0.5)",  "particle_color": "#80c8ff"},
        "starlight": {"glow": "rgba(160, 180, 255, 0.5)",  "particle_color": "#c8d8ff"},
        "growth":    {"glow": "rgba(100, 200, 80,  0.5)",  "particle_color": "#90e080"},
    }
    style = essence_styles.get(animation, essence_styles["fireice"])
    glow = style["glow"]
    particle_color = style["particle_color"]

    # Speaking state changes animation speed and glow intensity
    speak_filter = f"drop-shadow(0 0 32px {glow}) drop-shadow(0 0 12px {glow}) brightness(1.3)" if speaking else f"drop-shadow(0 0 24px {glow}) drop-shadow(0 0 8px {glow})"
    float_duration = "2s" if speaking else "4s"

    if use_video:
        media_tag = f"""
        <video
            class="spirit-media"
            src="data:video/mp4;base64,{media_b64}"
            autoplay loop muted playsinline
        ></video>
        """
    else:
        media_tag = f"""
        <img
            class="spirit-media"
            src="data:image/png;base64,{media_b64}"
            alt="{spirit['name']}"
        />
        """

    html_content = f"""
    <style>
    @keyframes spirit-float {{
        0%   {{ transform: scale(1) translateY(0px);   }}
        50%  {{ transform: scale(1.03) translateY(-8px); }}
        100% {{ transform: scale(1) translateY(0px);   }}
    }}
    @keyframes particle-float {{
        0%   {{ transform: translateY(0px) translateX(0px); opacity: 0; }}
        20%  {{ opacity: 0.7; }}
        80%  {{ opacity: 0.3; }}
        100% {{ transform: translateY(-50px) translateX(15px); opacity: 0; }}
    }}

    .spirit-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: transparent;
        padding: 8px 0 4px 0;
        margin-bottom: 4px;
        position: relative;
    }}
    .spirit-media {{
        width: 300px;
        height: 300px;
        object-fit: cover;
        border-radius: 50%;
        mask-image: radial-gradient(circle closest-side, black 75%, transparent 100%);
        -webkit-mask-image: radial-gradient(circle closest-side, black 75%, transparent 100%);
        animation: spirit-float {float_duration} ease-in-out infinite;
        position: relative;
        z-index: 2;
        filter: {speak_filter};
        transition: filter 0.6s ease;
    }}
    .spirit-name {{
        color: rgba(255,255,255,0.4);
        font-size: 10px;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 12px;
        font-family: Georgia, serif;
    }}
    .spirit-status {{
        color: rgba(255,255,255,0.2);
        font-size: 10px;
        margin-top: 4px;
        letter-spacing: 1px;
        font-style: italic;
    }}
    .particle {{
        position: absolute;
        width: 3px;
        height: 3px;
        border-radius: 50%;
        background: {particle_color};
        animation: particle-float linear infinite;
        pointer-events: none;
        z-index: 3;
    }}
    </style>

    <div class="spirit-container">
        <div class="particle" style="left:18%; bottom:20%; animation-duration:3.5s; animation-delay:0s;   opacity:0.5;"></div>
        <div class="particle" style="left:72%; bottom:15%; animation-duration:4.2s; animation-delay:1s;   opacity:0.45;"></div>
        <div class="particle" style="left:50%; bottom:5%;  animation-duration:3.8s; animation-delay:0.5s; opacity:0.6;"></div>
        <div class="particle" style="left:28%; bottom:35%; animation-duration:5s;   animation-delay:2s;   opacity:0.3;"></div>
        <div class="particle" style="left:80%; bottom:38%; animation-duration:4.5s; animation-delay:1.5s; opacity:0.4;"></div>
        <div class="particle" style="left:62%; bottom:22%; animation-duration:3.2s; animation-delay:0.8s; opacity:0.35;"></div>
        <div class="particle" style="left:38%; bottom:12%; animation-duration:4.8s; animation-delay:2.5s; opacity:0.3;"></div>

        {media_tag}
        <div class="spirit-name">{spirit['name']}</div>
        <div class="spirit-status">{"✦ present with you ✦" if not speaking else "✦ speaking ✦"}</div>
    </div>
    """

    st.components.v1.html(html_content, height=360)


# ==========================================
# SPIRIT QUESTIONNAIRE
# ==========================================
def render_questionnaire():
    st.markdown("### 🌌 Discover Your Kindred Spirit")
    st.markdown(
        "Answer from the heart — there's no right or wrong. "
        "Your spirit will be created from who you are, not just how you feel today."
    )
    st.divider()

    with st.form("spirit_questionnaire"):
        st.markdown("**How do you tend to move through the world?**")
        vibe = st.select_slider(
            "Your natural pace",
            options=["Gentle", "Quiet", "Peaceful", "Warm", "Hopeful", "Strong", "Restless", "Mysterious"],
            value="Gentle",
            label_visibility="collapsed",
        )

        st.markdown("**Which element feels most like home to you?**")
        element = st.radio(
            "Your element",
            options=["☁️ Fire&Ice", "💧 Water", "⭐ Starlight", "🌿 Growth"],
            horizontal=True,
            label_visibility="collapsed",
        )

        st.markdown("**What do you most need from a quiet, safe space?**")
        need = st.select_slider(
            "Your need",
            options=["Comfort", "Gentleness", "Peace", "Clarity", "Hope", "Strength"],
            value="Peace",
            label_visibility="collapsed",
        )

        st.markdown("**How do you experience your emotions?**")
        emotion_style = st.radio(
            "Emotion style",
            options=[
                "Like waves — they rise and fall",
                "Like light — they shift and change",
                "Like roots — they run deep and steady",
                "Like weather — they pass through me",
            ],
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button(
            "✨ Create My Spirit", use_container_width=True, type="primary"
        )

        if submitted:
            spirit_key = ESSENCE_MAP.get(element, "fireice")
            save_spirit(spirit_key)
            st.session_state.show_questionnaire = False
            st.success(f"Your spirit has been created — {SPIRITS[spirit_key]['name']}.")
            st.rerun()

    if st.button("← Back", use_container_width=True):
        st.session_state.show_questionnaire = False
        st.rerun()


# ==========================================
# HELPERS
# ==========================================
def auto_speak(text: str, speed: float) -> None:
    safe_text = html.escape(text, quote=True)
    st.components.v1.html(f"""
    <script>
    (function() {{
      if (!("speechSynthesis" in window)) return;
      window.speechSynthesis.cancel();
      var u = new SpeechSynthesisUtterance("{safe_text}");
      u.lang = "en-ZA";
      u.rate = {float(speed)};
      setTimeout(function() {{ window.speechSynthesis.speak(u); }}, 400);
    }})();
    </script>
    """, height=0)


def show_replay_button(text: str, speed: float, key: str) -> None:
    safe_text = html.escape(text, quote=True)
    st.components.v1.html(f"""
    <div id="kr-{key}">
      <span style="display:none">{safe_text}</span>
      <button onclick="(function(){{
        var t=document.querySelector('#kr-{key} span').textContent;
        window.speechSynthesis.cancel();
        var u=new SpeechSynthesisUtterance(t);
        u.lang='en-ZA'; u.rate={float(speed)};
        window.speechSynthesis.speak(u);
      }})()"
        style="background:transparent;border:0;color:#aaa;cursor:pointer;
               font-size:12px;padding:2px 4px;">
        🔊 Replay
      </button>
    </div>
    """, height=28)


def call_bedrock_stream(system_prompt: str, messages: list, placeholder, max_tokens: int = 400) -> str:
    bedrock_messages = [
        {"role": m["role"], "content": [{"text": m["content"]}]}
        for m in messages if m["role"] in ("user", "assistant")
    ]
    last_exc = None
    for model_id in MODELS:
        try:
            response = bedrock.converse_stream(
                modelId=model_id,
                system=[{"text": system_prompt}],
                messages=bedrock_messages,
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.7},
            )
            full_response = ""
            for event in response["stream"]:
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"]["delta"].get("text", "")
                    full_response += delta
                    placeholder.markdown(full_response + "▌")
            return full_response
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ThrottlingException":
                last_exc = exc
                continue
            raise
    raise last_exc


def call_bedrock(system_prompt: str, messages: list, max_tokens: int = 700) -> str:
    bedrock_messages = [
        {"role": m["role"], "content": [{"text": m["content"]}]}
        for m in messages if m["role"] in ("user", "assistant")
    ]
    last_exc = None
    for model_id in MODELS:
        try:
            response = bedrock.converse(
                modelId=model_id,
                system=[{"text": system_prompt}],
                messages=bedrock_messages,
                inferenceConfig={"maxTokens": max_tokens, "temperature": 0.7},
            )
            return response["output"]["message"]["content"][0]["text"]
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "ThrottlingException":
                last_exc = exc
                continue
            raise
    raise last_exc


def get_error_message(exc: Exception) -> str:
    if isinstance(exc, ClientError):
        code = exc.response["Error"]["Code"]
        if code == "AccessDeniedException":
            return "Access was denied. Check IAM permissions and AWS region."
        if code == "ThrottlingException":
            return "Too many requests. Please wait a moment and try again."
        if code in ("ValidationException", "ModelNotReadyException"):
            return "The AI model isn't available right now. Please try again."
    return "Kindred couldn't get a response just now. Please try again shortly."


def create_whisper(prompt: str) -> str:
    return call_bedrock(
        system_prompt=MAIN_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=180,
    )


def premium_gate(feature_name: str):
    st.info(
        f"✨ **{feature_name}** is a Premium feature. "
        "Upgrade to Kindred Premium (R39/month) to unlock this and more."
    )


# ==========================================
# HEADER
# ==========================================
st.title("🕊️ Kindred")

is_guest = st.session_state.user == "guest"
free_remaining = remaining_free_messages()
col_status, col_logout = st.columns([4, 1])
with col_status:
    if is_guest:
        st.caption("👤 Guest — free access")
    elif st.session_state.is_premium:
        st.caption(f"✨ Premium — {st.session_state.user.email}")
    else:
        st.caption(f"🤍 Free — {st.session_state.user.email}")
    if free_remaining is not None:
        st.caption(
            f"{free_remaining} of {FREE_DAILY_MESSAGE_LIMIT} free messages left today "
            "(resets at midnight SAST)."
        )
with col_logout:
    if st.button("Log out" if not is_guest else "Exit", use_container_width=True):
        logout()


# ==========================================
# QUESTIONNAIRE PAGE
# ==========================================
if st.session_state.show_questionnaire:
    render_questionnaire()
    st.stop()


# ==========================================
# SPIRIT DISPLAY (Premium only — shown above chat)
# ==========================================
if st.session_state.is_premium:
    if st.session_state.spirit_key:
        render_spirit(
            st.session_state.spirit_key,
            speaking=st.session_state.spirit_speaking
        )
        col_spirit1, col_spirit2 = st.columns([3, 1])
        with col_spirit2:
            if st.button("✦ Update Spirit", use_container_width=True):
                st.session_state.show_questionnaire = True
                st.rerun()
    else:
        # Premium user hasn't created their spirit yet
        st.markdown(
            """
            <div style='background:#0a0a0f; border-radius:20px; padding:32px;
                        text-align:center; margin-bottom:12px;'>
                <div style='font-size:48px; margin-bottom:12px;'>🌌</div>
                <div style='color:rgba(255,255,255,0.7); font-size:16px; margin-bottom:8px;'>
                    Your Kindred Spirit is waiting
                </div>
                <div style='color:rgba(255,255,255,0.4); font-size:13px;'>
                    Answer a few gentle questions to create your personal spirit
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("✨ Create My Spirit", use_container_width=True, type="primary"):
            st.session_state.show_questionnaire = True
            st.rerun()


# ==========================================
# MODE SELECTOR
# ==========================================
mode_options = ["🕊️ Kindred (Ages 13+)"]
if st.session_state.is_premium:
    mode_options.append("🌱 Kindred Kids (Up to age 12)")

mode_choice = st.radio(
    "Choose your space",
    options=mode_options,
    horizontal=True,
    label_visibility="collapsed",
)

new_mode = "kids" if "Kids" in mode_choice else "main"
if new_mode != st.session_state.mode:
    st.session_state.mode = new_mode
    st.session_state.messages = []
    st.session_state.auto_speak_text = ""
    st.rerun()

if st.session_state.mode == "main":
    st.subheader("A space to say what's on your mind")
    SYSTEM_PROMPT = MAIN_PROMPT
else:
    st.subheader("A friendly place to learn and grow")
    st.info("I'll help you understand — but I won't do the work for you. You've got this!")
    SYSTEM_PROMPT = KIDS_PROMPT


# ==========================================
# DISPLAY CONVERSATION
# ==========================================
for index, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_replay_button(msg["content"], st.session_state.voice_speed, f"h{index}")


# ==========================================
# VOICE INPUT
# ==========================================
chat_capped = free_remaining is not None and free_remaining <= 0
if chat_capped:
    render_free_cap_notice()
else:
    st.markdown("🎤 **Voice** — click Start, speak, click Stop when done")

voice_transcript = None
if not chat_capped:
    voice_transcript = speech_to_text(
        language="en-ZA",
        start_prompt="🎤 Start speaking",
        stop_prompt="⏹ Done — send",
        just_once=True,
        use_container_width=True,
        key="kindred_stt",
    )


# ==========================================
# CHAT INPUT AND RESPONSE
# ==========================================
user_input = st.chat_input(
    "Or type what's on your mind..."
    if st.session_state.mode == "main"
    else "Or type your question here",
    disabled=chat_capped,
)

used_voice = False
if voice_transcript and not user_input:
    user_input = voice_transcript
    used_voice = True

if user_input:
    if not consume_free_message():
        render_free_cap_notice()
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Activate spirit speaking state
        if st.session_state.spirit_key:
            st.session_state.spirit_speaking = True

        recent_messages = st.session_state.messages[-20:]

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            try:
                full_response = call_bedrock_stream(
                    system_prompt=SYSTEM_PROMPT,
                    messages=recent_messages,
                    placeholder=response_placeholder,
                    max_tokens=400,
                )
                if full_response.strip():
                    response_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    show_replay_button(full_response, st.session_state.voice_speed, "latest")
                    if used_voice:
                        st.session_state.auto_speak_text = full_response
                else:
                    response_placeholder.empty()
                    st.warning("The AI returned an empty response. Please try again.")
            except Exception as exc:
                response_placeholder.empty()
                st.error(get_error_message(exc))
            finally:
                # Return spirit to idle after response
                st.session_state.spirit_speaking = False

if st.session_state.auto_speak_text:
    auto_speak(st.session_state.auto_speak_text, st.session_state.voice_speed)
    st.session_state.auto_speak_text = ""


# ==========================================
# OPTIONAL FEATURES
# ==========================================
st.divider()

with st.expander("🔊 Voice Settings"):
    st.session_state.voice_speed = st.slider(
        "Speaking Speed",
        min_value=0.5,
        max_value=1.5,
        value=float(st.session_state.voice_speed),
        step=0.1,
        key="voice_speed_slider",
    )
    st.info("Click 🔊 Replay under any message to hear it again.")

with st.expander("✨ Daily Kindred Whisper", expanded=False):
    st.markdown("Take a moment. Breathe. Answer from your heart — there's no right or wrong.")

    vibe = st.select_slider(
        "What best describes your spirit today?",
        options=["Gentle", "Quiet", "Strong", "Restless", "Hopeful", "Weary", "Warm", "Mysterious"],
        value="Gentle",
    )
    element = st.selectbox(
        "Which element feels most like home to you?",
        ["Moonlight", "Ancient Forest", "Deep Ocean", "Soft Clouds", "Starlight", "Warm Glow"],
    )
    feeling = st.select_slider(
        "What do you need most right now?",
        options=["Peace", "Strength", "Comfort", "Hope", "Clarity", "Gentleness"],
        value="Peace",
    )

    if st.button("Receive Your Whisper", type="primary", use_container_width=True):
        if not consume_free_message():
            render_free_cap_notice()
        else:
            whisper_prompt = (
                f"Create a short, poetic message for someone who feels {vibe}, "
                f"connected to {element}, and seeking {feeling}. Keep it to 3-4 "
                "sentences maximum. Start with a short, meaningful name for this feeling. "
                "Be warm and comforting."
            )
            with st.spinner("Your whisper is taking shape..."):
                try:
                    st.session_state.spirit_text = create_whisper(whisper_prompt)
                    st.session_state.spirit_generated = True
                except Exception as exc:
                    st.error(get_error_message(exc))

    if st.session_state.spirit_generated and st.session_state.spirit_text:
        st.markdown("---")
        st.markdown(st.session_state.spirit_text)
        st.markdown("---")

        if st.session_state.is_premium:
            st.download_button(
                label="Save Your Whisper",
                data=st.session_state.spirit_text,
                file_name="my-kindred-whisper.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.caption("✨ Save & revisit whispers with Premium.")

        if st.button("New Whisper", use_container_width=True):
            st.session_state.spirit_generated = False
            st.session_state.spirit_text = ""
            st.rerun()

with st.expander("📄 Conversation History", expanded=False):
    if st.session_state.is_premium:
        st.info("Conversation saving coming soon.")
    else:
        premium_gate("Conversation History")


# ==========================================
# FOOTER
# ==========================================
st.divider()
st.caption(
    "Kindred uses AI to respond. Messages are processed securely via AWS. "
    "Free use includes a gentle daily message cap. "
    "Avoid sharing sensitive personal information. Children should use this app "
    "with guidance from a trusted adult."
)
