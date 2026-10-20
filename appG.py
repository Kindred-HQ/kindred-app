import html
import io

import boto3
import streamlit as st
from botocore.exceptions import ClientError
from streamlit_mic_recorder import mic_recorder, speech_to_text
from supabase import create_client, Client


# ==========================================
# PAGE SETUP
# ==========================================
st.set_page_config(page_title="Kindred", page_icon="🕊️", layout="centered")


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
    "user": None,          # Supabase user object
    "is_premium": False,
    "auth_view": "login",  # "login" | "signup"
    "auth_error": "",
    "auth_success": "",
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ==========================================
# AUTH HELPERS
# ==========================================
def get_profile(user_id: str) -> dict:
    """Fetch the user's profile row from Supabase."""
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
        st.session_state.auth_error = ""
        st.rerun()
    except Exception as e:
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
    for key in ["user", "is_premium", "messages", "spirit_generated", "spirit_text", "auto_speak_text"]:
        st.session_state[key] = defaults[key]
    st.rerun()


# ==========================================
# AUTH UI
# ==========================================
def render_auth():
    st.title("🕊️ Kindred")
    st.markdown("##### A gentle space to say what's on your mind.")
    st.divider()

    # Anonymous free access option
    st.markdown("**Continue without an account**")
    st.caption("Free access — chat anonymously, no sign-up needed.")
    if st.button("Enter as Guest", use_container_width=True):
        st.session_state.user = "guest"
        st.session_state.is_premium = False
        st.rerun()

    st.divider()

    # Toggle between login and signup
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
    """Show an upgrade prompt for gated features."""
    st.info(
        f"✨ **{feature_name}** is a Premium feature. "
        "Upgrade to Kindred Premium (R39/month) to unlock this and more."
    )


# ==========================================
# HEADER
# ==========================================
st.title("🕊️ Kindred")

# User status bar
is_guest = st.session_state.user == "guest"
col_status, col_logout = st.columns([4, 1])
with col_status:
    if is_guest:
        st.caption("👤 Guest — free access")
    elif st.session_state.is_premium:
        st.caption(f"✨ Premium — {st.session_state.user.email}")
    else:
        st.caption(f"🤍 Free — {st.session_state.user.email}")
with col_logout:
    if st.button("Log out" if not is_guest else "Exit", use_container_width=True):
        logout()


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
st.markdown("🎤 **Voice** — click Start, speak, click Stop when done")

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
    else "Or type your question here"
)

used_voice = False
if voice_transcript and not user_input:
    user_input = voice_transcript
    used_voice = True

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

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
        whisper_prompt = (
            f"Create a short, poetic message for someone who feels {vibe}, "
            f"connected to {element}, and seeking {feeling}. Keep it to 2-3 "
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

        # Save whisper — premium only
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

# Kindred Spirit — premium only
with st.expander("🌌 Your Kindred Spirit", expanded=False):
    if st.session_state.is_premium:
        st.info("Spirit creation coming soon — your personal, evolving ethereal companion.")
    else:
        premium_gate("Kindred Spirit")

# Conversation history — premium only
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
    "Avoid sharing sensitive personal information. Children should use this app "
    "with guidance from a trusted adult."
)
