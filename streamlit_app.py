import streamlit as st
import torch
import whisper
import spacy
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from fpdf import FPDF
from docx import Document
from pydub import AudioSegment

# --------- Cache the models ---------
@st.cache_resource
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("small", device=device)

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_sci_md")

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# ---------- Load models ----------
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

whisper_model = load_whisper_model()
nlp = load_spacy_model()
llm = load_llm()

# ---------- Streamlit App ----------

if "patient_db" not in st.session_state:
    st.session_state.patient_db = {}

patient_db = st.session_state.patient_db

st.title("🩺 AI-Powered Medical Documentation System")
st.write("Upload an audio file of a patient consultation to generate a structured medical report.")

# ---------- Patient Information (Registration) ----------

st.subheader("Patient Information")

pid = st.text_input("Enter Patient ID")
name = st.text_input("Enter Patient Name")
age = st.number_input("Enter Patient Age", min_value=0, max_value=150, step=1)
sex = st.selectbox("Select Sex", ("Male", "Female", "Other"))
register_button = st.button("Register Patient")

if register_button:
    if pid and name:
        patient_db[pid] = {
            "Name": name,
            "Age": age,
            "Sex": sex
        }
        st.success(f"Patient {name} registered successfully!")
    else:
        st.error("Please enter both Patient ID and Name.")

# ---------- Upload and Transcribe Audio ----------

st.subheader("Upload Audio Consultation")

audio_file = st.file_uploader("Upload an audio file...", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        # Important: Re-read the bytes properly for pydub
        audio = AudioSegment.from_file(audio_file)  # <-- FIXED ✅
        audio.export(tmp.name, format="wav")
        tmp_file_path = tmp.name

    with st.spinner("Transcribing..."):
        result = whisper_model.transcribe(tmp_file_path)
        transcription = result["text"]

    st.subheader("Transcription")
    st.write(transcription)

    # ---------- Extract Medical Entities ----------

    st.subheader("Extract Medical Information")

    with st.spinner("Extracting entities..."):
        doc = nlp(transcription)

    problems = [ent.text for ent in doc.ents if ent.label_ == "PROBLEM"]
    treatments = [ent.text for ent in doc.ents if ent.label_ == "TREATMENT"]
    tests = [ent.text for ent in doc.ents if ent.label_ == "TEST"]

    st.write("**Problems:**", problems)
    st.write("**Treatments:**", treatments)
    st.write("**Tests:**", tests)

    # ---------- Generate Final Report ----------

    st.subheader("Generate Medical Report")

    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            prompt = f"""
            Create a detailed medical report based on the following information:

            Patient ID: {pid}
            Name: {name}
            Age: {age}
            Sex: {sex}

            Consultation Details:
            {transcription}

            Extracted Problems: {problems}
            Extracted Treatments: {treatments}
            Extracted Tests: {tests}

            Format the report in a formal medical style.
            """
            response = llm.invoke([HumanMessage(content=prompt)])
            report = response.content

        st.subheader("Generated Report")
        st.write(report)

        # ---------- Download as PDF ----------

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        for line in report.split("\n"):
            pdf.multi_cell(0, 10, line)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            pdf.output(tmp_pdf.name)
            st.download_button("Download Report as PDF", data=open(tmp_pdf.name, "rb"), file_name="medical_report.pdf")

        # ---------- Download as Word Document ----------

        docx_file = Document()
        docx_file.add_heading("Medical Report", 0)
        docx_file.add_paragraph(report)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_docx:
            docx_file.save(tmp_docx.name)
            st.download_button("Download Report as Word Document", data=open(tmp_docx.name, "rb"), file_name="medical_report.docx")
