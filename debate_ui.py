import streamlit as st
from debate_back import generate_answer 

st.set_page_config(page_title="Debate & Research Assistant", layout="wide")

st.title("🗣️ Debate Research Assistant")
st.write(
    "Type your motion, select the debate format, team, and speaker role, then generate the debate case."
)

# -----------------------------
# USER INPUTS
# -----------------------------
motion = st.text_area("Enter the debate motion:", height=100)

debate_format = st.radio(
    "Select Debate Format:",
    ["British Parliamentary (BP)", "Asian Parliamentary (AP)"]
)

team = st.selectbox(
    "Select Team:",
    ["Government", "Opposition"]
)

speaker_role = st.selectbox(
    "Select Speaker Role:",
    [
        "Opening Speaker",
        "Reply Speaker",
        "Member of the Government/Opposition",
        "Other"
    ]
)

# -----------------------------
# GENERATE BUTTON
# -----------------------------
if st.button("Generate Debate Case"):
    if not motion.strip():
        st.warning("Please enter a motion before generating.")
    else:
        # Construct a structured query to send to backend
        user_query = f"""
Motion: {motion}
Debate Format: {debate_format}
Team: {team}
Speaker Role: {speaker_role}
"""

        with st.spinner("Generating your debate case..."):
            try:
                # Call your backend LLM function
                response = generate_answer(user_query)
                
                # Display output in markdown for formatting
                st.markdown("### 📝 Generated Debate Case")
                st.markdown(response, unsafe_allow_html=False)
            except Exception as e:
                st.error(f"Error generating response: {e}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("Developed by Sahil Bhatti | Powered by Mistral AI & LangChain")