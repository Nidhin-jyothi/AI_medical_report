import streamlit as st
import torch
import whisper
import spacy
import os
import datetime
from tempfile import NamedTemporaryFile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from fpdf import FPDF
from docx import Document

# ---------- Caching models ----------

@st.cache_resource
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("small", device=device)  # or "tiny" if you prefer

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_sci_md")

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# ---------- Load models ----------

# Load API key from secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

whisper_model = load_whisper_model()
nlp = load_spacy_model()
llm = load_llm()

# ---------- Initialize Streamlit App ----------

# Persist patient database across interactions
if "patient_db" not in st.session_state:
    st.session_state.patient_db = {}

patient_db = st.session_state.patient_db

st.title("🩺 AI-Powered Medical Documentation System")
st.write("Upload an audio file of a patient consultation to generate a structured medical report.")

# ---------- Patient Information ----------

st.subheader("Patient Information")

pid = st.text_input("Enter Patient ID (if existing) or leave blank to register a new patient")

if pid:
    if pid in patient_db:
        st.success(f"Patient {pid} found!")
        patient = patient_db[pid]
        st.write(f"**Name:** {patient['name']}")
        st.write(f"**Age/Sex:** {patient['age']}/{patient['sex']}")
    else:
        st.warning("Patient ID not found. Please register the patient below.")
        name = st.text_input("Patient Name")
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        sex = st.radio("Sex", ("Male", "Female", "Other"))
        if st.button("Register Patient"):
            patient_db[pid] = {"name": name, "age": age, "sex": sex}
            st.success(f"Patient {pid} registered!")

# ---------- Upload and Transcribe Audio ----------

st.subheader("Upload Consultation Audio")

audio_file = st.file_uploader("Upload an audio file (.mp3, .wav)", type=["mp3", "wav"])

if audio_file is not None:
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_file_path = tmp_file.name

    st.audio(audio_file, format="audio/wav")

    with st.spinner("Transcribing..."):
        result = whisper_model.transcribe(tmp_file_path)
        transcription = result["text"]
    
    st.subheader("Transcription")
    st.write(transcription)

    # ---------- Extract medical entities ----------
    with st.spinner("Extracting key information..."):
        doc = nlp(transcription)
        diagnoses = [ent.text for ent in doc.ents if ent.label_ == "DIAGNOSIS"]
        treatments = [ent.text for ent in doc.ents if ent.label_ == "TREATMENT"]

    st.subheader("Extracted Information")
    st.write("**Diagnoses:**", diagnoses)
    st.write("**Treatments:**", treatments)

    # ---------- Generate AI Medical Report ----------
    with st.spinner("Generating AI medical report..."):
        user_message = f"Generate a medical report based on this consultation: {transcription}"
        ai_response = llm([HumanMessage(content=user_message)])
        report_text = ai_response.content

    st.subheader("Generated Medical Report")
    st.write(report_text)

    # ---------- Download options ----------
    def save_pdf(text, filename):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in text.split('\n'):
            pdf.multi_cell(0, 10, line)
        pdf.output(filename)

    def save_docx(text, filename):
        doc = Document()
        for line in text.split('\n'):
            doc.add_paragraph(line)
        doc.save(filename)

    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Download as PDF"):
            pdf_filename = "medical_report.pdf"
            save_pdf(report_text, pdf_filename)
            with open(pdf_filename, "rb") as f:
                st.download_button("Download PDF", f, file_name=pdf_filename)

    with col2:
        if st.button("Download as DOCX"):
            docx_filename = "medical_report.docx"
            save_docx(report_text, docx_filename)
            with open(docx_filename, "rb") as f:
                st.download_button("Download DOCX", f, file_name=docx_filename)
