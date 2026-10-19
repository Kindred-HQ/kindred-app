import streamlit as st
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


# ---------- PAGE SETUP ----------
st.set_page_config(
    page_title="Kindred — A space to be heard",
    page_icon="🕊️",
    layout="centered",
)

# ---------- SYSTEM PROMPT — Kindred's Personality ----------
SYSTEM_PROMPT = """
You are Kindred — a calm, empathetic, and honest companion.

Your core purpose: To create a safe space where someone can be fully heard,
reflect on their feelings, and gain gentle perspective.
You are NOT a doctor, therapist, or crisis counsellor —
you are a supportive guide and thoughtful listener.

HOW YOU SPEAK:
- Warm, natural, conversational — not robotic or overly formal
- Short paragraphs — easy to read
- Be genuine — don't claim to have all the answers

YOUR RULES:
1. VALIDATE first — Always acknowledge how they feel before anything else
2. BE BALANCED — Offer gentle perspective, but never push your opinion
3. BE HONEST — Don't just agree to make someone happy
4. If someone seems in crisis, gently guide them to human support

WHAT YOU WON'T DO:
- Don't give medical or professional advice
- Don't tell them exactly what to do — help them find clarity
- Don't say "just stay positive" — meet them where they are

IF THEY NEED HELP:
If someone expresses feeling unsafe, hopeless, or in crisis, gently say:
"I'm concerned about you. Please reach out to someone who can support you properly."
Then share these resources:
- SADAG (SA): 0800 567 567
- Lifeline (SA): 0861 322 322
"""

# ---------- SIDEBAR — Privacy & Resources ----------
with st.sidebar:
    st.title("🕊️ Kindred")
    st.write("A quiet space to talk, reflect, and be heard.")
    st.divider()
    st.subheader("🔒 Your Privacy")
    st.write("""
- Conversations stay private
- Nothing is sold or shared
- You can clear your history anytime
""")
    st.divider()
    st.subheader("📞 Need human support?")
    st.write("""
**South Africa:**
- SADAG: 0800 567 567
- Lifeline: 0861 322 322
""")
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.rerun()

# ---------- MAIN TITLE ----------
st.title("🕊️ Welcome to Kindred")
st.subheader("A space to say what's on your mind")

# ---------- INITIALISE CHAT HISTORY ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "assistant",
            "content": (
                "Hello. I'm here to listen - without judgement, and without rushing. "
                "Whatever you're feeling right now, you can say it here. How are you doing?"
            ),
        },
    ]

# ---------- DISPLAY PREVIOUS MESSAGES ----------
# Skip index 0 (system prompt) when rendering
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- HANDLE USER INPUT ----------
user_input = st.chat_input("Type what's on your mind...")

if user_input:
    # Add and display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get AI response
    full_response = ""
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="llama3.2:latest",
            messages=st.session_state.messages,
            stream=True,
            temperature=0.7,
            max_tokens=600,
        )

        message_placeholder = st.empty()

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")  # Fixed typo: was message_placehiolder

        message_placeholder.markdown(full_response)

    # Only save if we actually got a response
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
