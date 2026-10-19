import html
import time

import streamlit as st
from openai import OpenAI, APIError, AuthenticationError, RateLimitError


# ==========================================
# PAGE SETUP
# ==========================================
st.set_page_config(page_title="Kindred", page_icon="🕊️", layout="centered")
st.title("🕊️ Welcome to Kindred")


# ==========================================
# OPENROUTER CONFIGURATION
# ==========================================
# Primary model with fallbacks — if the first is rate-limited, the next is tried.
MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "meta-llama/llama-3.3-8b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
]

# You must define OPENROUTER_API_KEY in Streamlit secrets.
try:
    api_key = st.secrets["OPENROUTER_API_KEY"]
except Exception:
    st.error(
        "OpenRouter API key is missing. Add OPENROUTER_API_KEY to "
        "your Streamlit secrets before using Kindred."
    )
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    timeout=60.0,
    max_retries=0,  # Handle rate limits ourselves; avoid hidden repeat requests.
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


def get_error_message(exc: Exception) -> str:
    """Return a user-friendly message without exposing credentials or raw payloads."""
    details = str(exc).lower()

    if isinstance(exc, AuthenticationError):
        return (
            "OpenRouter rejected the API key. Check OPENROUTER_API_KEY "
            "in Streamlit secrets."
        )

    if isinstance(exc, RateLimitError):
        if any(term in details for term in ("quota", "credit", "insufficient", "balance")):
            return (
                "The AI provider reports a quota or credit limit. Check your "
                "OpenRouter activity, key limits, and account credits."
            )
        return (
            "Kindred has reached a request limit, or the selected free model "
            "is temporarily busy. Wait a little and try again. If this continues, "
            "check OpenRouter activity and the model's current limits."
        )

    return (
        "Kindred couldn't get a response from the AI service. Please try again "
        "shortly. If it continues, check your OpenRouter activity and app logs."
    )


def create_whisper(prompt: str) -> str:
    """Request a short daily whisper from the model, with fallbacks."""
    last_exc = None
    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": MAIN_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=180,
            )
            return response.choices[0].message.content or "Take a slow breath. You can begin again gently."
        except RateLimitError as exc:
            last_exc = exc
            time.sleep(1)
            continue
    raise last_exc


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
# CHAT INPUT AND RESPONSE
# ==========================================
user_input = st.chat_input(
    "Type what's on your mind..."
    if st.session_state.mode == "main"
    else "What would you like to learn today? 😊"
)

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Keep the prompt and recent conversation to limit context growth.
    recent_messages = st.session_state.messages[-20:]
    messages_for_model = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages_for_model.extend(
        {"role": message["role"], "content": message["content"]}
        for message in recent_messages
        if message["role"] in ("user", "assistant")
    )

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            # Try each model in the fallback list until one succeeds.
            last_exc = None
            for model in MODELS:
                try:
                    stream = client.chat.completions.create(
                        model=model,
                        messages=messages_for_model,
                        stream=True,
                        temperature=0.7,
                        max_tokens=700,
                    )
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                            response_placeholder.markdown(full_response + "▌")
                    last_exc = None
                    break  # Success — stop trying further models
                except RateLimitError as exc:
                    last_exc = exc
                    full_response = ""  # Reset before retrying next model
                    time.sleep(1)
                    continue
            if last_exc:
                raise last_exc

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
                st.warning("The AI service returned an empty response. Please try again.")

        except (RateLimitError, AuthenticationError, APIError) as exc:
            response_placeholder.empty()
            st.error(get_error_message(exc))
        except Exception:
            response_placeholder.empty()
            st.error(
                "Something unexpected interrupted the response. Please try again. "
                "If this continues, check your Streamlit app logs."
            )


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
            f"connected to {element}, and seeking {feeling}. Keep it to 2–3 "
            "sentences maximum. Start with a short, meaningful name for this feeling. "
            "Be warm and comforting."
        )
        with st.spinner("Your whisper is taking shape..."):
            try:
                st.session_state.spirit_text = create_whisper(whisper_prompt)
                st.session_state.spirit_generated = True
            except (RateLimitError, AuthenticationError, APIError) as exc:
                st.error(get_error_message(exc))
            except Exception:
                st.error("Your whisper couldn't be generated just now. Please try again.")

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
    "🕊️ Kindred uses AI to respond. Messages are sent to OpenRouter for processing. "
    "Avoid sharing sensitive personal information. Children should use this app "
    "with guidance from a trusted adult."
)