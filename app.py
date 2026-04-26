import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Multi-Input Text Summarizer",
    page_icon="📝",
    layout="centered"
)

st.title("📝 AI-Powered Multi-Input Summarizer")
st.markdown("Combine up to five distinct text inputs and generate a comprehensive, concise summary.")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    summary_length = st.selectbox(
        "Summary Length",
        ("Short (~3 sentences)", "Medium (1 paragraph)", "Long (Detailed summary)"),
        index=1
    )
    
    # Map selection to descriptive instruction
    length_instruction = {
        "Short (~3 sentences)": "a very concise summary of about 3 sentences",
        "Medium (1 paragraph)": "a clear and concise paragraph (5–7 lines max)",
        "Long (Detailed summary)": "a detailed summary covering all key aspects in multiple concise paragraphs"
    }[summary_length]
    


# Callback to clear forms.
def clear_inputs():
    for i in range(1, 6):
        if f"input_{i}" in st.session_state:
            st.session_state[f"input_{i}"] = ""

# Input Section
st.subheader("Enter your texts below:")

inputs = []
for i in range(1, 6):
    val = st.text_area(f"Enter Text {i}", key=f"input_{i}", height=100)
    inputs.append(val)

col1, col2 = st.columns([1, 1])
with col1:
    summarize_clicked = st.button("🚀 Summarize", use_container_width=True)
with col2:
    st.button("🗑️ Clear Form", on_click=clear_inputs, use_container_width=True)

st.divider()

# Summarize Logic
if summarize_clicked:
    # Check if at least one field is filled
    has_text = any(text.strip() for text in inputs)
    
    if not has_text:
        st.error("⚠️ Please enter text in at least one field to generate a summary.")
    else:
        # Check API key
        if not os.getenv("GROQ_API_KEY"):
            st.error("⚠️ System configuration error. Please check backend settings.")
            st.stop()
        
        with st.spinner("🧠 Generating summary "):
            try:
                # Initialize ChatGroq LLM
                llm = ChatGroq(
                    model="llama-3.1-8b-instant", 
                    temperature=0.3, # low temp for consistent summarization
                )
                
                # Combine all five inputs into a structured prompt
                prompt_text = f"""You are an intelligent summarization assistant.

Your task is to analyze multiple input texts and generate a highly concise, well-structured summary.

### Instructions:

* Combine the inputs into a single coherent summary.
* Do NOT simply merge or rephrase sentences — extract key insights.
* Length requirement: {length_instruction}.
* Avoid repetition, filler words, and unnecessary transitions.
* Maintain clarity and logical flow.

### Special Rule for Healthcare Content:

If the input contains medical, health, or wellness-related information:

* Use a slightly more careful and informative tone.
* Do NOT provide diagnosis, treatment, or medical advice.
* Only summarize the information safely and neutrally.
* If needed, include a soft disclaimer like:
  "This is a general summary and not medical advice."

---

### Input:

Text 1: {{input1}}
Text 2: {{input2}}
Text 3: {{input3}}
Text 4: {{input4}}
Text 5: {{input5}}

---

### Output:

Provide only the final summary paragraph."""
                
                # Set up Langchain Prompt
                prompt = PromptTemplate(
                    input_variables=["input1", "input2", "input3", "input4", "input5"],
                    template=prompt_text
                )
                
                # Chain setup using LCEL
                chain = prompt | llm | StrOutputParser()
                
                # Run the LLM
                response = chain.invoke({
                    "input1": inputs[0],
                    "input2": inputs[1],
                    "input3": inputs[2],
                    "input4": inputs[3],
                    "input5": inputs[4]
                })
                
                # Display output
                st.subheader("✨ Generated Summary")
                st.success(response)
                
            except Exception as e:
                st.error(f"❌ An error occurred during summarization.\n\nDetails: {e}")
