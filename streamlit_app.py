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
import subprocess
import sys

@st.cache_resource
subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_sci_md"])

# ---------- Caching models ----------

@st.cache_resource
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("small", device=device)  # small or tiny

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
            st.success(f"Patient {pid} registered successfully!")

# ---------- Audio Upload ----------

st.subheader("Upload Consultation Audio")

audio_file = st.file_uploader("Upload an audio file (wav, mp3, m4a)", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_file_path = tmp_file.name

    with st.spinner("Transcribing..."):
        result = whisper_model.transcribe(tmp_file_path)
        transcription = result["text"]

    st.subheader("Transcription")
    st.write(transcription)

    # ---------- Extract Medical Entities ----------

    with st.spinner("Extracting medical terms..."):
        doc = nlp(transcription)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

    st.subheader("Extracted Medical Entities")
    for ent_text, ent_label in entities:
        st.write(f"• {ent_text} ({ent_label})")

    # ---------- Generate Medical Report ----------

    with st.spinner("Generating report..."):
        response = llm.invoke([HumanMessage(content=f"Generate a detailed medical report based on the following consultation transcript:\n\n{transcription}")])
        report_text = response.content

    st.subheader("Generated Medical Report")
    st.write(report_text)

    # ---------- Download Report ----------

    def save_as_pdf(text, filename):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for line in text.split('\n'):
            pdf.multi_cell(0, 10, line)
        pdf.output(filename)

    def save_as_docx(text, filename):
        doc = Document()
        for line in text.split('\n'):
            doc.add_paragraph(line)
        doc.save(filename)

    st.subheader("Download Report")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Download PDF"):
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                save_as_pdf(report_text, tmp_pdf.name)
                st.download_button("Download PDF", tmp_pdf.read(), file_name="medical_report.pdf")

    with col2:
        if st.button("Download Word"):
            with NamedTemporaryFile(delete=False, suffix=".docx") as tmp_docx:
                save_as_docx(report_text, tmp_docx.name)
                st.download_button("Download Word", tmp_docx.read(), file_name="medical_report.docx")
