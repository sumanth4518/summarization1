import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="AI Multi-Input Summarizer",
    page_icon="📝",
    layout="centered"
)

st.title("📝 AI-Powered Multi-Input Summarizer")
st.markdown("Generate a structured, meaningful summary from multiple inputs more accurately.")

# Sidebar
with st.sidebar:
    st.header("Settings")

    summary_length = st.selectbox(
        "Summary Length",
        ("Short", "Medium", "Long"),
        index=1
    )

    if summary_length == "Short":
        length_instruction = "Keep it concise with 3–5 bullet points and a short paragraph (3–4 lines)."

    elif summary_length == "Medium":
        length_instruction = "Provide 5–8 detailed bullet points and a well-structured paragraph (5–7 lines)."

    else:
        length_instruction = """Provide 8–12 detailed bullet points covering all inputs thoroughly.
Also include a detailed explanation in 2 well-structured paragraphs (each 5–7 lines)."""

# Clear inputs
def clear_inputs():
    for i in range(1, 6):
        st.session_state[f"input_{i}"] = ""

# Inputs
st.subheader("Enter your texts below:")

inputs = []
for i in range(1, 6):
    val = st.text_area(f"Enter Text {i}", key=f"input_{i}", height=100)
    inputs.append(val)

col1, col2 = st.columns(2)

with col1:
    summarize_clicked = st.button("🚀 Summarize", use_container_width=True)

with col2:
    st.button("🗑️ Clear Form", on_click=clear_inputs, use_container_width=True)

st.divider()

# Main Logic
if summarize_clicked:

    cleaned_inputs = [text.strip() for text in inputs if text.strip()]

    if not cleaned_inputs:
        st.error("⚠️ Please enter at least one text input.")
        st.stop()

    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ GROQ API key not found. Check your .env file.")
        st.stop()

    with st.spinner("🧠 Generating structured summary..."):
        try:
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.2
            )

            combined_inputs = "\n".join(
                [f"Text {i+1}: {txt}" for i, txt in enumerate(cleaned_inputs)]
            )

            prompt_text = f"""
You are an expert AI summarization assistant.

Your goal is to analyze multiple inputs and produce a structured summary.

---

### STRICT OUTPUT FORMAT:

## 📌 Combined Summary

### 🔹 Key Insights
- Cover ALL important points from ALL inputs
- Each bullet must be meaningful and unique
- Do NOT skip any input

### 🔹 Consolidated Understanding
Follow the length instruction strictly:
{length_instruction}

---

### RULES:

- Do NOT copy sentences directly
- Extract insights and combine intelligently
- Avoid repetition
- Keep it clear and structured
- Do NOT add any information not present in inputs

---

### Healthcare Rule (STRICT):

Check if the inputs are clearly related to healthcare.

- If YES → Add EXACTLY this line at the end:
"This is a general summary and not medical advice."

- If NO → DO NOT include any disclaimer.

---

### INPUTS:
{combined_inputs}

---

### OUTPUT:
Follow the format strictly.
"""

            prompt = PromptTemplate(
                input_variables=[],
                template=prompt_text
            )

            chain = prompt | llm | StrOutputParser()

            response = chain.invoke({})

            # 🔥 FINAL FIX: Remove unwanted disclaimer completely
            response = "\n".join(
                line for line in response.split("\n")
                if "not medical advice" not in line.lower()
            ).strip()

            st.subheader("✨ Generated Summary")
            st.markdown(response)

        except Exception as e:
            st.error(f"❌ Error: {e}")
