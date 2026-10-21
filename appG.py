import base64
import hashlib
import html
import io
import json
import logging
import os
import random
import statistics
from datetime import datetime, timedelta, timezone
from typing import Optional

import boto3
import replicate
import streamlit as st
from botocore.exceptions import ClientError
from PIL import Image
from PIL import ImageChops
from PIL import ImageFilter
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
        background-color: #b4b8f0;
    }
    .stApp {
        --background-color: #b4b8f0;
        --text-color: #242333;
        color: #242333;
    }
    [data-testid="stHeader"] {
        background-color: rgba(180, 184, 240, 0.92);
    }
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] {
        background-color: #b4b8f0 !important;
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
SPIRIT_STORAGE_BUCKET = "kindred-spirits"


@st.cache_resource
def get_supabase_admin() -> Client:
    service_role_key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY")
    if not service_role_key:
        raise RuntimeError(
            "Add SUPABASE_SERVICE_ROLE_KEY to Streamlit secrets before creating a spirit."
        )
    return create_client(st.secrets["SUPABASE_URL"], service_role_key)


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
REPLICATE_IMAGE_MODEL = "black-forest-labs/flux-schnell"
REPLICATE_BACKGROUND_MODEL = "cjwbw/rembg"


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

# Spirit image prompt rules
ELEMENT_RULES = {
    "Fire": {
        "base_colors": ["crimson", "warm amber", "soft orange-gold"],
        "base_form": "luminous flame-like flowing abstract shape",
        "base_movement": "flows and drifts upward continuously",
    },
    "Water": {
        "base_colors": ["teal", "soft sky-blue", "silver glow"],
        "base_form": "ribbon-like rippling abstract shape",
        "base_movement": "flows and curves gracefully",
    },
    "Starlight": {
        "base_colors": ["deep indigo", "star-silver", "pale periwinkle"],
        "base_form": "cloud-like stardust abstract shape",
        "base_movement": "expands and drifts outward steadily",
    },
    "Growth": {
        "base_colors": ["soft moss-green", "honey-gold", "deep forest-teal"],
        "base_form": "root-and-sprout flowing abstract shape",
        "base_movement": "breathes and unfurls slowly",
    },
    "Air-Cosmic": {
        "base_colors": ["pale lavender", "pale cyan", "moon-white"],
        "base_form": "swirling weightless mist abstract shape",
        "base_movement": "twirls and drifts freely",
    },
}

PACE_RULES = {
    "Gentle": {"speed": "slow, soft-paced", "motion_mod": "smooth long continuous flowing trails"},
    "Steady": {"speed": "calm, consistent pace", "motion_mod": "steady continuous pulsing rhythm"},
    "Lively": {"speed": "bright, moderately quick", "motion_mod": "lighter continuous swirling flow"},
    "Quiet": {"speed": "very slow, subtle", "motion_mod": "barely drifting, soft faint continuous pulses"},
}

STRENGTH_RULES = {
    "Soft-open": {"texture": "wispy, translucent, mist-like", "energy_tone": "open, tender presence"},
    "Resilient-steady": {"texture": "dense glowing core with soft outer flow", "energy_tone": "stable, enduring presence"},
    "Bright-passionate": {"texture": "luminous, vivid glowing edges", "energy_tone": "alive, heartfelt presence"},
    "Wise-deep": {"texture": "layered, deep-glowing mist", "energy_tone": "calm, knowing presence"},
}

QUALITY_RULES = {
    "Comfort": {"accent_detail": "soft rounded outward-flowing form"},
    "Clarity": {"accent_detail": "clean defined glowing flowing lines"},
    "Freedom": {"accent_detail": "wide outward-reaching continuous flow"},
    "Safety": {"accent_detail": "centered, gently outward-wrapping flow"},
}

STYLE_SEED_RULES = {
    "seed_01": {"global_style": "smooth minimal ethereal"},
    "seed_02": {"global_style": "smoky, soft-blended ethereal"},
    "seed_03": {"global_style": "crystalline, sharp-glow ethereal"},
    "seed_04": {"global_style": "wispy delicate ethereal"},
    "seed_05": {"global_style": "luminous radiant ethereal"},
}

def build_spirit_prompt(elem, pace, strength, quality, seed_key):
    try:
        element_rules = ELEMENT_RULES[elem]
        pace_rules = PACE_RULES[pace]
        strength_rules = STRENGTH_RULES[strength]
        quality_rules = QUALITY_RULES[quality]
        style_rules = STYLE_SEED_RULES[seed_key]
    except KeyError as exc:
        raise ValueError(f"Unknown spirit design option: {exc.args[0]}") from exc

    return (
        "Create an ethereal abstract spirit image, later to become a 5-second seamless forward-only loop animation.\n"
        f"Core form: {element_rules['base_form']}, made of glowing mist and light in {', '.join(element_rules['base_colors'])}.\n"
        f"Movement intent: {element_rules['base_movement']}, at {pace_rules['speed']}, with {pace_rules['motion_mod']}.\n"
        f"Texture and energy: {strength_rules['texture']}, {strength_rules['energy_tone']}.\n"
        f"Accent feature: {quality_rules['accent_detail']}.\n"
        f"Overall style: {style_rules['global_style']}, dream-like, soft glow.\n"
        "Composition: one centered floating abstract presence, isolated against a pure black background. Use continuous one-direction flow, designed for a seamless loop with matching start and end frames."
    )


SPIRIT_NEGATIVE_PROMPT = (
    "circle, border, frame, human, face, animal, text, lettering, typography, "
    "icon, logo, multiple subjects"
)


def remove_spirit_background(image_base64: str) -> str:
    image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGBA")
    rgb_image = image.convert("RGB")
    width, height = image.size
    inset = max(8, min(32, min(width, height) // 64))
    sample_size = min(
        64,
        max(16, min(width, height) // 32),
        width - 2 * inset,
        height - 2 * inset,
    )
    corner_origins = (
        (inset, inset),
        (width - inset - sample_size, inset),
        (inset, height - inset - sample_size),
        (width - inset - sample_size, height - inset - sample_size),
    )
    corner_profiles = []
    for origin_x, origin_y in corner_origins:
        samples = [
            rgb_image.getpixel((x, y))
            for x in range(origin_x, origin_x + sample_size)
            for y in range(origin_y, origin_y + sample_size)
        ]
        color = tuple(
            int(statistics.median(sample[channel] for sample in samples))
            for channel in range(3)
        )
        uniformity = statistics.median(
            max(abs(sample[channel] - color[channel]) for channel in range(3))
            for sample in samples
        )
        corner_profiles.append((color, uniformity))

    def background_score(profile):
        color, uniformity = profile
        cluster_size = sum(
            max(abs(other[channel] - color[channel]) for channel in range(3)) <= 32
            for other, _ in corner_profiles
        )
        canvas_distance = max(
            abs(color[channel] - (180, 184, 240)[channel])
            for channel in range(3)
        )
        return cluster_size, -uniformity, -canvas_distance

    background_color = max(corner_profiles, key=background_score)[0]
    background = Image.new("RGB", image.size, background_color)
    red_difference, green_difference, blue_difference = ImageChops.difference(
        rgb_image, background
    ).split()
    color_distance = ImageChops.lighter(
        ImageChops.lighter(red_difference, green_difference), blue_difference
    )
    color_alpha = color_distance.point(
        lambda distance: min(255, max(0, (distance - 24) * 255 // 48))
    )
    dark_alpha = rgb_image.convert("L").point(
        lambda brightness: min(255, max(0, (brightness - 48) * 255 // 48))
    )
    background_mask = ImageChops.darker(color_alpha, dark_alpha).filter(
        ImageFilter.GaussianBlur(radius=0.8)
    )
    image.putalpha(ImageChops.multiply(image.getchannel("A"), background_mask))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def generate_spirit_image(prompt: str) -> str:
    if not 1 <= len(prompt) <= 10000:
        raise ValueError("Spirit image prompt must be between 1 and 10000 characters.")

    api_token = st.secrets.get("REPLICATE_API_TOKEN")
    if not api_token:
        raise RuntimeError(
            "Add REPLICATE_API_TOKEN to Streamlit secrets before creating a spirit."
        )

    client = replicate.Client(api_token=api_token)
    generated_outputs = client.run(
        REPLICATE_IMAGE_MODEL,
        input={
            "prompt": f"{prompt}\nAvoid: {SPIRIT_NEGATIVE_PROMPT}.",
            "aspect_ratio": "1:1",
            "output_format": "png",
            "num_outputs": 1,
            "num_inference_steps": 4,
        },
    )
    generated_output = next(iter(generated_outputs), None)
    if generated_output is None:
        raise RuntimeError("Replicate returned no spirit image.")
    generated_bytes = generated_output.read()

    removed_background = client.run(
        REPLICATE_BACKGROUND_MODEL,
        input={"image": io.BytesIO(generated_bytes)},
    )
    if isinstance(removed_background, (list, tuple)):
        removed_background = removed_background[0] if removed_background else None
    if removed_background is None or not hasattr(removed_background, "read"):
        raise RuntimeError("Replicate returned no background-removed spirit image.")
    return base64.b64encode(removed_background.read()).decode("ascii")

# Temporary legacy display mapping until generated spirit media is connected.
ESSENCE_MAP = {
    "Fire": "fireice",
    "Water": "water",
    "Starlight": "starlight",
    "Growth": "growth",
    "Air-Cosmic": "fireice",
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
    "spirit_configuration": None,
    "spirit_prompt": "",
    "spirit_style_seed": None,
    "spirit_image": None,
    "spirit_image_processed": 0,
    "spirit_storage_path": None,
    "spirit_created_at": None,
    "spirit_load_error": False,
    "spirit_speaking": False,  # true while Kindred is responding
    "show_questionnaire": False,
    "guest_usage_id": None,
    "usage_date": "",
    "usage_count": 0,
    "scroll_to_top_after_entry": False,
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
        st.session_state.spirit_storage_path = profile.get("spirit_storage_path")
        st.session_state.spirit_created_at = profile.get("spirit_created_at")
        st.session_state.spirit_configuration = profile.get("spirit_configuration")
        st.session_state.spirit_style_seed = (
            (st.session_state.spirit_configuration or {}).get("style_seed")
        )
        st.session_state.spirit_image = None
        st.session_state.spirit_image_processed = 0
        st.session_state.spirit_load_error = False
        if st.session_state.spirit_storage_path:
            try:
                image_bytes = get_supabase_admin().storage.from_(
                    SPIRIT_STORAGE_BUCKET
                ).download(st.session_state.spirit_storage_path)
                st.session_state.spirit_image = base64.b64encode(
                    image_bytes
                ).decode("ascii")
                st.session_state.spirit_image_processed = 3
            except Exception:
                st.session_state.spirit_load_error = True
        st.session_state.usage_date = ""
        st.session_state.usage_count = 0
        st.session_state.auth_error = ""
        st.session_state.scroll_to_top_after_entry = True
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
            "guest_usage_id", "usage_date", "usage_count", "spirit_configuration",
                "spirit_prompt", "spirit_style_seed", "spirit_image",
                "spirit_image_processed",
                "spirit_storage_path", "spirit_created_at", "spirit_load_error"]:
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


def save_generated_spirit(user_id: str, spirit_key: str, image_base64: str,
                          configuration: dict) -> tuple:
    admin = get_supabase_admin()
    storage = admin.storage.from_(SPIRIT_STORAGE_BUCKET)
    previous_path = get_profile(user_id).get("spirit_storage_path")
    created_at = datetime.now(timezone.utc)
    storage_path = f"{user_id}/{created_at.strftime('%Y%m%dT%H%M%S%fZ')}.png"
    image_bytes = base64.b64decode(image_base64, validate=True)

    storage.upload(
        storage_path,
        image_bytes,
        file_options={"content-type": "image/png", "upsert": "false"},
    )
    try:
        result = (
            admin.table("profiles")
            .update({
                "spirit_key": spirit_key,
                "spirit_storage_path": storage_path,
                "spirit_created_at": created_at.isoformat(),
                "spirit_configuration": configuration,
            })
            .eq("id", user_id)
            .select("id")
            .execute()
        )
        if not result.data:
            raise RuntimeError("Could not save the spirit to the user profile.")
    except Exception:
        storage.remove([storage_path])
        raise

    if previous_path and previous_path != storage_path:
        try:
            storage.remove([previous_path])
        except Exception:
            pass

    return storage_path, created_at.isoformat()


def verify_spirit_storage_ready() -> None:
    admin = get_supabase_admin()
    admin.table("profiles").select(
        "spirit_storage_path,spirit_created_at,spirit_configuration"
    ).limit(1).execute()
    admin.storage.get_bucket(SPIRIT_STORAGE_BUCKET)


def spirit_regeneration_days_remaining() -> int:
    created_at = st.session_state.get("spirit_created_at")
    if not created_at:
        return 0
    try:
        created_at = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
    except ValueError:
        return 0

    remaining_seconds = (
        created_at + timedelta(days=30) - datetime.now(timezone.utc)
    ).total_seconds()
    if remaining_seconds <= 0:
        return 0
    return int((remaining_seconds + 86399) // 86400)


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
        st.session_state.scroll_to_top_after_entry = True
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

    generated_image = st.session_state.get("spirit_image")
    if generated_image and st.session_state.spirit_image_processed != 3:
        generated_image = remove_spirit_background(generated_image)
        st.session_state.spirit_image = generated_image
        st.session_state.spirit_image_processed = 3
    if generated_image:
        use_video = False
        media_b64 = generated_image
        spirit_name = "Your Kindred Spirit"
    else:
        use_video = os.path.exists(video_path)
        media_path = video_path if use_video else fallback_path
        if not os.path.exists(media_path):
            return
        with open(media_path, "rb") as f:
            media_b64 = base64.b64encode(f.read()).decode()
        spirit_name = spirit["name"]

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
        container_class = "spirit-container"
    else:
        media_class = "spirit-media spirit-generated" if generated_image else "spirit-media"
        container_class = (
            "spirit-container spirit-generated-container"
            if generated_image
            else "spirit-container"
        )
        media_tag = f"""
        <img
            class="{media_class}"
            src="data:image/png;base64,{media_b64}"
            alt="{spirit_name}"
        />
        """

    html_content = f"""
    <style>
    html, body {{
        background: transparent !important;
        margin: 0;
        padding: 0;
    }}
    @keyframes spirit-float {{
        0%   {{ transform: scale(1) translateY(0px);   }}
        50%  {{ transform: scale(1.03) translateY(-8px); }}
        100% {{ transform: scale(1) translateY(0px);   }}
    }}
    @keyframes spirit-generated-drift {{
        0%   {{ transform: translateY(1px) rotate(0deg) scale(1); }}
        30%  {{ transform: translateY(-3px) rotate(-0.35deg) scale(1.015); }}
        68%  {{ transform: translateY(-9px) rotate(0.3deg) scale(1.035); }}
        100% {{ transform: translateY(1px) rotate(0deg) scale(1); }}
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
    .spirit-media.spirit-generated {{
        width: 217px;
        height: 217px;
        object-fit: contain;
        border-radius: 0;
        mask-image: none;
        -webkit-mask-image: none;
        mix-blend-mode: normal;
        animation: spirit-generated-drift 4.5s ease-in-out infinite;
        filter: saturate(1.25) contrast(1.12);
        filter: url(#spirit-flow) saturate(1.25) contrast(1.12);
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

    <svg aria-hidden="true" width="0" height="0" style="position:absolute">
        <filter id="spirit-flow" x="-12%" y="-12%" width="124%" height="124%">
            <feTurbulence
                type="fractalNoise"
            baseFrequency="0.010 0.022"
                numOctaves="2"
                seed="7"
                result="flow-noise"
            >
                <animate
                    attributeName="baseFrequency"
                    values="0.010 0.022;0.005 0.012;0.010 0.022"
                    dur="6s"
                    repeatCount="indefinite"
                />
            </feTurbulence>
            <feDisplacementMap
                in="SourceGraphic"
                in2="flow-noise"
                scale="15"
                xChannelSelector="R"
                yChannelSelector="G"
            />
        </filter>
    </svg>

    <div class="{container_class}">
        <div class="particle" style="left:18%; bottom:20%; animation-duration:3.5s; animation-delay:0s;   opacity:0.5;"></div>
        <div class="particle" style="left:72%; bottom:15%; animation-duration:4.2s; animation-delay:1s;   opacity:0.45;"></div>
        <div class="particle" style="left:50%; bottom:5%;  animation-duration:3.8s; animation-delay:0.5s; opacity:0.6;"></div>
        <div class="particle" style="left:28%; bottom:35%; animation-duration:5s;   animation-delay:2s;   opacity:0.3;"></div>
        <div class="particle" style="left:80%; bottom:38%; animation-duration:4.5s; animation-delay:1.5s; opacity:0.4;"></div>
        <div class="particle" style="left:62%; bottom:22%; animation-duration:3.2s; animation-delay:0.8s; opacity:0.35;"></div>
        <div class="particle" style="left:38%; bottom:12%; animation-duration:4.8s; animation-delay:2.5s; opacity:0.3;"></div>

        {media_tag}
        <div class="spirit-name">{spirit_name}</div>
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
        st.markdown("**Which element feels most like home to you?**")
        element = st.radio(
            "Your element",
            options=list(ELEMENT_RULES),
            horizontal=True,
            label_visibility="collapsed",
        )

        st.markdown("**What pace feels most like you?**")
        pace = st.selectbox(
            "Your pace",
            options=list(PACE_RULES),
            label_visibility="collapsed",
        )

        st.markdown("**Which quality feels closest to your inner strength?**")
        strength = st.selectbox(
            "Your strength",
            options=list(STRENGTH_RULES),
            label_visibility="collapsed",
        )

        st.markdown("**What would you most like your spirit to offer?**")
        quality = st.selectbox(
            "Your spirit's quality",
            options=list(QUALITY_RULES),
            label_visibility="collapsed",
        )

        days_remaining = spirit_regeneration_days_remaining()
        if days_remaining:
            st.info(f"You can renew your spirit in {days_remaining} days.")
        st.caption(
            "Creating a spirit makes two paid Replicate requests: image generation "
            "and background removal."
        )
        submitted = st.form_submit_button(
            "✨ Create My Spirit" if not st.session_state.spirit_created_at
            else "✨ Renew My Spirit",
            use_container_width=True,
            type="primary",
            disabled=days_remaining > 0,
        )

        if submitted:
            if days_remaining:
                st.info(f"You can renew your spirit in {days_remaining} days.")
                return
            seed_key = random.choice(list(STYLE_SEED_RULES))
            spirit_key = ESSENCE_MAP[element]
            configuration = {
                "element": element,
                "pace": pace,
                "strength": strength,
                "quality": quality,
                "style_seed": seed_key,
            }
            spirit_prompt = build_spirit_prompt(
                element, pace, strength, quality, seed_key
            )
            with st.spinner("Creating your spirit image..."):
                try:
                    verify_spirit_storage_ready()
                    generated_image = generate_spirit_image(spirit_prompt)
                    storage_path, created_at = save_generated_spirit(
                        st.session_state.user.id,
                        spirit_key,
                        generated_image,
                        configuration,
                    )
                    st.session_state.spirit_image = generated_image
                    st.session_state.spirit_image_processed = 3
                    st.session_state.spirit_configuration = configuration
                    st.session_state.spirit_style_seed = seed_key
                    st.session_state.spirit_prompt = spirit_prompt
                    st.session_state.spirit_storage_path = storage_path
                    st.session_state.spirit_created_at = created_at
                    st.session_state.spirit_load_error = False
                    st.session_state.show_questionnaire = False
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))
                except Exception:
                    logging.exception("Replicate spirit generation or save failed")
                    st.error(
                        "Could not create or save your spirit. Check the Replicate token "
                        "and Supabase storage setup; see the app logs for details."
                    )

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

st.tabs(
    mode_options,
    default=(
        "🌱 Kindred Kids (Up to age 12)"
        if st.session_state.mode == "kids" and st.session_state.is_premium
        else mode_options[0]
    ),
    key="kindred_mode_tabs",
    on_change="rerun",
)
mode_choice = st.session_state["kindred_mode_tabs"]

new_mode = "kids" if "Kids" in mode_choice else "main"
if new_mode != st.session_state.mode:
    st.session_state.mode = new_mode
    st.session_state.messages = []
    st.session_state.auto_speak_text = ""

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

if st.session_state.pop("scroll_to_top_after_entry", False):
    st.components.v1.html(
        """
        <script>
        function scrollAppToTop() {
            try {
                const parentWindow = window.parent;
                parentWindow.scrollTo(0, 0);
                const appView = parentWindow.document.querySelector(
                    '[data-testid="stAppViewContainer"]'
                );
                const mainView = parentWindow.document.querySelector(
                    '[data-testid="stMain"]'
                );
                if (appView) appView.scrollTo(0, 0);
                if (mainView) mainView.scrollTo(0, 0);
            } catch (error) {
                window.parent.scrollTo(0, 0);
            }
        }
        window.setTimeout(scrollAppToTop, 100);
        window.setTimeout(scrollAppToTop, 600);
        </script>
        """,
        height=0,
    )
