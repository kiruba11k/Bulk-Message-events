import streamlit as st
import pandas as pd
from typing import Optional, TypedDict
import os
from groq import Groq
from langgraph.graph import StateGraph, END
import re
from io import BytesIO

# ==========================
# CONFIGURATION & CLIENT
# ==========================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

class ProspectMessageState(TypedDict):
    prospect_name: Optional[str]
    designation: Optional[str]
    company: Optional[str]
    industry: Optional[str]
    prospect_background: str
    my_background: Optional[str]
    final_message: Optional[str]

# ==========================
# LLM CALL
# ==========================
def groq_llm(prompt: str, model: str = "llama3-8b-8192", temperature: float = 0.3) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

# ==========================
# SUMMARIZER
# ==========================
def summarizer(text: str) -> str:
    if not text or not isinstance(text, str):
        return "No content to summarize."
    truncated_text = text[:4000]
    prompt = f"""
Create 3 concise bullet points from this background text. Focus on key professional highlights and achievements:

{truncated_text}

Bullet points:
-"""
    try:
        return groq_llm(prompt).strip()
    except Exception:
        return "Background summary unavailable"

def summarize_backgrounds(state: ProspectMessageState) -> ProspectMessageState:
    return {**state, "prospect_background": summarizer(state["prospect_background"])}

# ==========================
# NAME EXTRACTION
# ==========================
def extract_name_from_background(background: str) -> str:
    if not background:
        return "there"
    match = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?', background)
    if match:
        return match[0]
    return "there"

# ==========================
# MESSAGE GENERATION
# ==========================
def generate_message(state: ProspectMessageState) -> ProspectMessageState:
    extracted_name = extract_name_from_background(state['prospect_background'])
    prospect_first_name = extracted_name.split()[0] if extracted_name != "Unknown Prospect" else "there"
    my_name = "Sumana"  

    clean_background = re.sub(
        r'at\s+\w+\s+(\w+\s+){0,3}(of|Systems|America|Inc\.?|Ltd\.?|Corp\.?)?',
        '',
        state['prospect_background'],
        flags=re.IGNORECASE
    )
    clean_background = re.sub(r'\s*specializing in\s*\w+|\s*with\s+\w+\s+experience|\s*as\s+a\s+\w+', '', clean_background)
    clean_background = re.sub(r'\s{2,}', ' ', clean_background).strip()

    prompt = f"""
IMPORTANT: Output ONLY the message itself.
Do NOT include any explanations, labels, or introductions.
Create a SHORT LinkedIn connection message (MAX 3 LINES , 250 chars) following this natural pattern:

Hi {prospect_first_name},
I see that you'll be attending {state.get('event_name', '')}. Highlight ONE specific achievement/expertise from their background without mentioning companies or job titles.
Close with "Best, {my_name}"

Prospect: {state['prospect_name']}
Key Highlight: {clean_background}
Event: {state.get('event_name', '')}

Message:
Hi {prospect_first_name},"""

    try:
        response = groq_llm(prompt, temperature=0.7)
        message = response.strip()

        # Clean unwanted prefixes
        unwanted_starts = ["Here is", "Here’s", "LinkedIn connection message:", "Message:", "Output:"]
        for phrase in unwanted_starts:
            if message.lower().startswith(phrase.lower()):
                message = message.split("\n", 1)[-1].strip()

        # Ensure closing signature
        if f"Best, {my_name}" not in message:
            message += f"\n\nBest, {my_name}"

        return {**state, "final_message": message}
    except Exception:
        return {**state, "final_message": "Failed to generate message"}

# Workflow graph
workflow = StateGraph(ProspectMessageState)
workflow.add_node("summarize_backgrounds", summarize_backgrounds)
workflow.add_node("generate_message", generate_message)
workflow.set_entry_point("summarize_backgrounds")
workflow.add_edge("summarize_backgrounds", "generate_message")
workflow.add_edge("generate_message", END)
graph1 = workflow.compile()

# ==========================
# STREAMLIT UI
# ==========================
st.set_page_config(page_title="LinkedIn Bulk Message Generator", layout="centered")
st.title("Bulk LinkedIn Message Generator")

uploaded_file = st.file_uploader("Upload CSV file with LinkedIn data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    required_cols = ["Name", "LinkedinData"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain columns: {', '.join(required_cols)}")
    else:
        st.success("CSV uploaded successfully!")
        if st.button("Generate Messages for All Prospects"):
            with st.spinner("Generating messages for all prospects..."):
                messages = []
                for idx, row in df.iterrows():
                    state = {
                        "prospect_name": row["prospect_name"],
                        "designation": row.get("designation", ""),
                        "company": row.get("company", ""),
                        "industry": row.get("industry", ""),
                        "prospect_background": row["prospect_background"],
                        "my_background": "",
                        "event_name": "Step San Francisco 2025",
                        "event_details": "August 12-14, MGM Grand Las Vegas"
                    }
                    result = graph1.invoke(state)
                    messages.append(result["final_message"])

                df["Generated_Message"] = messages

                # Convert to downloadable CSV
                output = BytesIO()
                df.to_csv(output, index=False)
                output.seek(0)

                st.success("Messages generated successfully!")
                st.download_button(
                    label="Download Updated CSV",
                    data=output,
                    file_name="linkedin_messages.csv",
                    mime="text/csv"
                )
