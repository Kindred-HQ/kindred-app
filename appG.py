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
# Cross-region inference profile IDs required for eu-west-1.
MODELS = [
    "eu.amazon.nova-lite-v1:0",
    "eu.amazon.nova-micro-v1:0",
]

# Load AWS credentials from Streamlit secrets.
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
- If they are stuck, say: "Let's try one step at a time together 😊"
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
if "enable_voice" not in st.session_state:
    st.session_state.enable_voice = False
if "voice_speed" not in st.session_state:
    st.session_state.voice_speed = 1.0
if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""


# ==========================================
# HELPERS
# ==========================================
def show_voice_button(text: str, speed: float, key: str) -> None:
    """Render a speech-synthesis button with text safely escaped for HTML."""
    safe_text = html.escape(text, quote=True)
    component_html = f"""
    <div id="kindred-voice-{key}">
      <span id="kindred-text" style="display:none">{safe_text}</span>
      <button id="kindred-speak"
        style="background:transparent;border:0;color:#888;cursor:pointer;
               font-size:14px;padding:4px 8px;">
        🔊 Speak
      </button>
      <script>
        const root = document.getElementById("kindred-voice-{key}");
        const button = root.querySelector("#kindred-speak");
        const text = root.querySelector("#kindred-text").textContent;
        button.addEventListener("click", () => {{
          if (!("speechSynthesis" in window) ||
              !("SpeechSynthesisUtterance" in window)) {{
            button.textContent = "Voice is not supported by this browser";
            return;
          }}
          window.speechSynthesis.cancel();
          const utterance = new SpeechSynthesisUtterance(text);
          utterance.rate = {float(speed)};
          window.speechSynthesis.speak(utterance);
        }});
      </script>
    </div>
    """
    st.components.v1.html(component_html, height=42)


def call_bedrock(system_prompt: str, messages: list, max_tokens: int = 700) -> str:
    """Call Bedrock non-streaming, with model fallback."""
    bedrock_messages = [
        {"role": msg["role"], "content": [{"text": msg["content"]}]}
        for msg in messages
        if msg["role"] in ("user", "assistant")
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
    """Stream a Bedrock response, updating placeholder live. Falls back on throttling."""
    bedrock_messages = [
        {"role": msg["role"], "content": [{"text": msg["content"]}]}
        for msg in messages
        if msg["role"] in ("user", "assistant")
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
    """Return a user-friendly error message without leaking raw details."""
    if isinstance(exc, ClientError):
        error_code = exc.response["Error"]["Code"]
        if error_code == "AccessDeniedException":
            return (
                "Access was denied. Make sure the IAM user has "
                "AmazonBedrockFullAccess and that the model is enabled in "
                "your AWS region (eu-west-1)."
            )
        if error_code == "ThrottlingException":
            return "Kindred is receiving too many requests right now. Please wait a moment and try again."
        if error_code in ("ValidationException", "ModelNotReadyException"):
            return "The AI model isn't available right now. Please try again in a moment."
    return "Kindred couldn't get a response just now. Please try again shortly."


def create_whisper(prompt: str) -> str:
    """Request a short daily whisper from Bedrock."""
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
    st.rerun()

if st.session_state.mode == "main":
    st.subheader("A space to say what's on your mind")
    SYSTEM_PROMPT = MAIN_PROMPT
else:
    st.subheader("✨ A friendly place to learn and grow")
    st.info("💡 I'll help you understand — but I won't do the work for you. You've got this!")
    SYSTEM_PROMPT = KIDS_PROMPT


# ==========================================
# DISPLAY CONVERSATION
# ==========================================
for index, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and st.session_state.enable_voice:
            show_voice_button(
                msg["content"],
                st.session_state.voice_speed,
                f"history-{index}",
            )


# ==========================================
# VOICE INPUT
# ==========================================
# Voice input uses a custom Streamlit component to capture speech.
# The transcript is passed back via st.query_params and submitted as a message.
voice_input_html = """
<style>
  #kindred-mic-wrap { display:flex; align-items:center; gap:10px; padding:4px 0 8px 0; }
  #kindred-mic-btn {
    background:#f0f2f6; border:1px solid #ccc; border-radius:20px;
    padding:6px 18px; cursor:pointer; font-size:14px; color:#444;
    transition: background 0.2s;
  }
  #kindred-mic-btn.listening { background:#ffe0e0; border-color:#f88; }
  #kindred-transcript {
    flex:1; padding:6px 12px; border:1px solid #ddd; border-radius:8px;
    font-size:14px; color:#333; background:#fff; min-width:0;
  }
  #kindred-send-btn {
    background:#5b8cff; color:white; border:none; border-radius:20px;
    padding:6px 18px; cursor:pointer; font-size:14px;
  }
  #kindred-send-btn:disabled { background:#bbb; cursor:default; }
  #kindred-mic-status { font-size:12px; color:#888; padding: 0 0 4px 2px; }
</style>
<div id="kindred-mic-wrap">
  <button id="kindred-mic-btn">🎤 Speak</button>
  <input id="kindred-transcript" type="text" placeholder="Your words will appear here..." readonly/>
  <button id="kindred-send-btn" disabled>Send ➤</button>
</div>
<div id="kindred-mic-status"></div>
<script>
(function() {
  const micBtn   = document.getElementById("kindred-mic-btn");
  const box      = document.getElementById("kindred-transcript");
  const sendBtn  = document.getElementById("kindred-send-btn");
  const status   = document.getElementById("kindred-mic-status");

  if (!("webkitSpeechRecognition" in window || "SpeechRecognition" in window)) {
    micBtn.textContent = "🎤 Not supported";
    micBtn.disabled = true;
    status.textContent = "Use Chrome or Edge for voice input.";
    return;
  }

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = new SR();
  recognition.lang = "en-ZA";
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;
  let listening = false;

  micBtn.addEventListener("click", () => {
    if (listening) { recognition.stop(); return; }
    box.value = "";
    sendBtn.disabled = true;
    recognition.start();
  });

  recognition.onstart = () => {
    listening = true;
    micBtn.textContent = "⏹ Stop";
    micBtn.classList.add("listening");
    status.textContent = "🎙 Listening...";
  };

  recognition.onend = () => {
    listening = false;
    micBtn.textContent = "🎤 Speak";
    micBtn.classList.remove("listening");
    status.textContent = box.value ? "✅ Ready — click Send" : "";
    if (box.value.trim()) sendBtn.disabled = false;
  };

  recognition.onresult = (event) => {
    let final = "", interim = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      if (event.results[i].isFinal) final += event.results[i][0].transcript;
      else interim += event.results[i][0].transcript;
    }
    box.value = final || interim;
  };

  recognition.onerror = (event) => {
    listening = false;
    micBtn.textContent = "🎤 Speak";
    micBtn.classList.remove("listening");
    status.textContent = "Mic error: " + event.error + ". Check browser mic permissions.";
  };

  sendBtn.addEventListener("click", () => {
    const text = box.value.trim();
    if (!text) return;
    // Pass transcript to Streamlit via URL query param, then trigger navigation
    const url = new URL(window.parent.location.href);
    url.searchParams.set("vc", encodeURIComponent(text));
    window.parent.location.href = url.toString();
  });
})();
</script>
"""

st.components.v1.html(voice_input_html, height=80)

# Pick up voice transcript from query params if present
vc_param = st.query_params.get("vc", "")
if vc_param and vc_param != st.session_state.voice_transcript:
    st.session_state.voice_transcript = vc_param
    # Clear the param so it doesn't re-fire on reruns
    st.query_params.clear()
    st.session_state.pending_voice_input = vc_param
    st.rerun()

# ==========================================
# CHAT INPUT AND RESPONSE
# ==========================================
user_input = st.chat_input(
    "Or type what's on your mind..."
    if st.session_state.mode == "main"
    else "Or type your question here 😊"
)

# Use voice input if available, otherwise typed input
if "pending_voice_input" in st.session_state and st.session_state.pending_voice_input:
    user_input = st.session_state.pending_voice_input
    st.session_state.pending_voice_input = ""

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
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
                if st.session_state.enable_voice:
                    show_voice_button(
                        full_response,
                        st.session_state.voice_speed,
                        "new-reply",
                    )
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
    st.session_state.enable_voice = st.checkbox(
        "🔊 Show Speak buttons under responses",
        value=st.session_state.enable_voice,
        key="voice_enabled_checkbox",
    )
    if st.session_state.enable_voice:
        st.session_state.voice_speed = st.slider(
            "Speaking Speed",
            min_value=0.5,
            max_value=1.5,
            value=float(st.session_state.voice_speed),
            step=0.1,
            key="voice_speed_slider",
        )
        st.info("💡 Click 🔊 Speak under a reply to hear it. Browser voice support may vary.")

with st.expander("✨ Daily Kindred Whisper", expanded=False):
    st.markdown("Take a moment. Breathe. Answer from your heart — there's no right or wrong.")

    vibe = st.select_slider(
        "🌿 What best describes your spirit today?",
        options=["Gentle", "Quiet", "Strong", "Restless", "Hopeful", "Weary", "Warm", "Mysterious"],
        value="Gentle",
    )
    element = st.selectbox(
        "🌌 Which element feels most like home to you?",
        ["🌙 Moonlight", "🌿 Ancient Forest", "🌊 Deep Ocean", "☁️ Soft Clouds", "✨ Starlight", "🔥 Warm Glow"],
    )
    feeling = st.select_slider(
        "💛 What do you need most right now?",
        options=["Peace", "Strength", "Comfort", "Hope", "Clarity", "Gentleness"],
        value="Peace",
    )

    if st.button("✨ Receive Your Whisper", type="primary", use_container_width=True):
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
            label="📄 Save Your Whisper",
            data=st.session_state.spirit_text,
            file_name="my-kindred-whisper.txt",
            mime="text/plain",
            use_container_width=True,
        )
        if st.button("🔄 New Whisper", use_container_width=True):
            st.session_state.spirit_generated = False
            st.session_state.spirit_text = ""
            st.rerun()


# ==========================================
# FOOTER
# ==========================================
st.divider()
st.caption(
    "🕊️ Kindred uses AI to respond. Messages are processed securely via AWS. "
    "Avoid sharing sensitive personal information. Children should use this app "
    "with guidance from a trusted adult."
)
