import html

import boto3
import streamlit as st
from botocore.exceptions import ClientError


# ==========================================
# PAGE SETUP
# ==========================================
st.set_page_config(page_title="Kindred", page_icon="🕊️", layout="centered")
st.title("🕊️ Welcome to Kindred")


# ==========================================
# AMAZON BEDROCK CONFIGURATION
# ==========================================
MODELS = [
    "eu.amazon.nova-lite-v1:0",
    "eu.amazon.nova-micro-v1:0",
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
- Keep responses concise but meaningful.
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
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "main"
if "spirit_generated" not in st.session_state:
    st.session_state.spirit_generated = False
if "spirit_text" not in st.session_state:
    st.session_state.spirit_text = ""
if "voice_speed" not in st.session_state:
    st.session_state.voice_speed = 1.0
if "last_spoken_response" not in st.session_state:
    st.session_state.last_spoken_response = ""


# ==========================================
# HELPERS
# ==========================================
def auto_speak(text: str, speed: float) -> None:
    """Immediately speak text using browser SpeechSynthesis — no button needed."""
    safe_text = html.escape(text, quote=True)
    st.components.v1.html(f"""
    <script>
      (function() {{
        if (!("speechSynthesis" in window)) return;
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance("{safe_text}");
        utterance.lang = "en-ZA";
        utterance.rate = {float(speed)};
        // Small delay ensures the page has settled before speaking
        setTimeout(() => window.speechSynthesis.speak(utterance), 300);
      }})();
    </script>
    """, height=0)


def show_voice_button(text: str, speed: float, key: str) -> None:
    """Render a manual 🔊 Speak button for replaying a response."""
    safe_text = html.escape(text, quote=True)
    st.components.v1.html(f"""
    <div id="kv-{key}">
      <span style="display:none">{safe_text}</span>
      <button onclick="(function(){{
        var t=document.querySelector('#kv-{key} span').textContent;
        window.speechSynthesis.cancel();
        var u=new SpeechSynthesisUtterance(t);
        u.lang='en-ZA'; u.rate={float(speed)};
        window.speechSynthesis.speak(u);
      }})()"
        style="background:transparent;border:0;color:#888;cursor:pointer;font-size:13px;padding:2px 6px;">
        🔊 Replay
      </button>
    </div>
    """, height=32)


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


def call_bedrock_stream(system_prompt: str, messages: list, placeholder, max_tokens: int = 700) -> str:
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


# ==========================================
# MODE SELECTOR
# ==========================================
mode_choice = st.radio(
    "Choose your space",
    options=["🕊️ Kindred (Ages 13+)", "🌱 Kindred Kids (Up to age 12)"],
    horizontal=True,
    index=0 if st.session_state.mode == "main" else 1,
    label_visibility="collapsed",
)

new_mode = "main" if "Kindred (Ages 13+)" in mode_choice else "kids"
if new_mode != st.session_state.mode:
    st.session_state.mode = new_mode
    st.session_state.messages = []
    st.session_state.last_spoken_response = ""
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
            show_voice_button(msg["content"], st.session_state.voice_speed, f"h{index}")


# ==========================================
# VOICE INPUT COMPONENT
# ==========================================
# Uses browser SpeechRecognition which auto-stops after ~1.5s of silence.
# When speech ends it posts the transcript to Streamlit via query param + page reload.
st.components.v1.html("""
<style>
  #km-wrap { display:flex; align-items:center; gap:8px; padding:6px 0; }
  #km-btn {
    border-radius:20px; border:1px solid #ccc; padding:6px 16px;
    cursor:pointer; font-size:14px; background:#f0f2f6; color:#444;
    transition:background .2s;
  }
  #km-btn.on { background:#ffe0e0; border-color:#f99; }
  #km-status { font-size:12px; color:#888; }
</style>
<div id="km-wrap">
  <button id="km-btn">🎤 Speak to Kindred</button>
  <span id="km-status"></span>
</div>
<script>
(function(){
  const btn = document.getElementById("km-btn");
  const status = document.getElementById("km-status");

  if (!("webkitSpeechRecognition" in window || "SpeechRecognition" in window)){
    btn.textContent = "🎤 Voice not supported (use Chrome/Edge)";
    btn.disabled = true;
    return;
  }

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const rec = new SR();
  rec.lang = "en-ZA";
  rec.interimResults = false;  // Only fire when confident — this enables auto-stop on silence
  rec.maxAlternatives = 1;
  let on = false;

  btn.addEventListener("click", () => {
    if (on) { rec.stop(); return; }
    rec.start();
  });

  rec.onstart = () => {
    on = true;
    btn.textContent = "🔴 Listening... (pausing will auto-send)";
    btn.classList.add("on");
    status.textContent = "";
  };

  rec.onend = () => {
    on = false;
    btn.textContent = "🎤 Speak to Kindred";
    btn.classList.remove("on");
  };

  rec.onresult = (e) => {
    const transcript = e.results[0][0].transcript.trim();
    if (!transcript) return;
    status.textContent = "Sending: " + transcript;
    // Pass to Streamlit via query param and reload
    const url = new URL(window.parent.location.href);
    url.searchParams.set("vc", transcript);
    window.parent.location.href = url.toString();
  };

  rec.onerror = (e) => {
    on = false;
    btn.textContent = "🎤 Speak to Kindred";
    btn.classList.remove("on");
    if (e.error !== "no-speech") status.textContent = "Error: " + e.error;
  };
})();
</script>
""", height=60)

# Pick up voice transcript from query params
vc_param = st.query_params.get("vc", "").strip()
if vc_param:
    st.query_params.clear()
    if "pending_vc" not in st.session_state or st.session_state.pending_vc != vc_param:
        st.session_state.pending_vc = vc_param

# ==========================================
# CHAT INPUT AND RESPONSE
# ==========================================
user_input = st.chat_input(
    "Or type what's on your mind..."
    if st.session_state.mode == "main"
    else "Or type your question here"
)

# Voice takes priority
if not user_input and st.session_state.get("pending_vc"):
    user_input = st.session_state.pending_vc
    st.session_state.pending_vc = ""
    used_voice = True
else:
    used_voice = False

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
            )
            if full_response.strip():
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                show_voice_button(full_response, st.session_state.voice_speed, "latest")
                # Auto-speak if user spoke to us
                if used_voice and full_response != st.session_state.last_spoken_response:
                    st.session_state.last_spoken_response = full_response
                    auto_speak(full_response, st.session_state.voice_speed)
            else:
                response_placeholder.empty()
                st.warning("The AI returned an empty response. Please try again.")
        except Exception as exc:
            response_placeholder.empty()
            st.error(get_error_message(exc))


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
    st.info("Use 🎤 Speak to Kindred above for hands-free conversation. "
            "Use 🔊 Replay under any message to hear it again.")

with st.expander("Daily Kindred Whisper", expanded=False):
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
        st.download_button(
            label="Save Your Whisper",
            data=st.session_state.spirit_text,
            file_name="my-kindred-whisper.txt",
            mime="text/plain",
            use_container_width=True,
        )
        if st.button("New Whisper", use_container_width=True):
            st.session_state.spirit_generated = False
            st.session_state.spirit_text = ""
            st.rerun()


# ==========================================
# FOOTER
# ==========================================
st.divider()
st.caption(
    "Kindred uses AI to respond. Messages are processed securely via AWS. "
    "Avoid sharing sensitive personal information. Children should use this app "
    "with guidance from a trusted adult."
)
