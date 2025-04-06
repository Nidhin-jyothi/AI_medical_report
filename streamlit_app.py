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

# ---------- Caching models ----------

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

# ---------- Patient Information ----------

st.subheader("Patient Information")

pid = st.text_input("Enter Patient ID (if existing) or leave blank to register a new patient")

if pid:
    if pid in patient_db:
        patient_info = patient_db[pid]
        st.success(f"Patient {pid} found.")
        st.write(patient_info)
    else:
        st.warning("Patient ID not found. Please register.")

else:
    st.info("New patient registration")
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=0, max_value=120)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    if st.button("Register Patient"):
        if name:
            new_pid = str(len(patient_db) + 1)
            patient_db[new_pid] = {"name": name, "age": age, "gender": gender}
            st.success(f"Patient registered successfully with ID: {new_pid}")
        else:
            st.error("Please enter the patient's name.")

# ---------- Audio Upload and Transcription ----------

st.subheader("Upload Consultation Audio")

audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name
        tmp.write(audio_file.read())  # ✅ properly write uploaded file content

    with st.spinner("Transcribing..."):
        result = whisper_model.transcribe(tmp_path)
        transcription = result["text"]

    st.subheader("Transcription")
    st.write(transcription)

    # ---------- Generate Medical Report ----------

    st.subheader("Generated Medical Report")

    prompt = f"""
    You are a medical scribe. Based on the following consultation transcription:

    {transcription}

    Generate a structured medical report with sections:
    - Chief Complaint
    - History of Present Illness
    - Past Medical History
    - Medications
    - Physical Examination
    - Assessment and Plan
    """

    with st.spinner("Generating Report..."):
        response = llm([HumanMessage(content=prompt)])
        report = response.content

    st.write(report)

    # ---------- Download Report ----------

    st.subheader("Download Report")

    def save_as_docx(text, filename):
        doc = Document()
        for line in text.split('\n'):
            doc.add_paragraph(line)
        doc.save(filename)

    def save_as_pdf(text, filename):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        for line in text.split('\n'):
            pdf.multi_cell(0, 10, line)
        pdf.output(filename)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Download as DOCX"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_docx:
                save_as_docx(report, tmp_docx.name)
                with open(tmp_docx.name, "rb") as f:
                    st.download_button(
                        label="Download DOCX",
                        data=f,
                        file_name="medical_report.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )

    with col2:
        if st.button("Download as PDF"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                save_as_pdf(report, tmp_pdf.name)
                with open(tmp_pdf.name, "rb") as f:
                    st.download_button(
                        label="Download PDF",
                        data=f,
                        file_name="medical_report.pdf",
                        mime="application/pdf",
                    )
