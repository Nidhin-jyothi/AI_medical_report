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

# Load SciSpaCy Model
nlp = spacy.load("en_core_sci_md")

# Load Whisper Model (small or tiny)
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small", device=device)  # you can change "small" to "tiny" if needed

model = load_whisper_model()

# Set up API Key securely from secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Persist patient database across interactions
if "patient_db" not in st.session_state:
    st.session_state.patient_db = {}

patient_db = st.session_state.patient_db

# Streamlit UI
st.title("🩺 AI-Powered Medical Documentation System")
st.write("Upload an audio file of a patient consultation to generate a structured medical report.")

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
            if name and age and sex:
                patient_db[pid] = {"name": name, "age": age, "sex": sex}
                st.success(f"Patient {pid} registered successfully!")

# Upload Audio
st.subheader("Upload Consultation Audio")
audio_file = st.file_uploader("Upload an audio file (.mp3, .wav)", type=["mp3", "wav"])

if audio_file is not None:
    st.audio(audio_file)
    with NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_filepath = tmp_file.name

    # Transcribe audio
    st.info("Transcribing audio... please wait ⏳")
    result = model.transcribe(tmp_filepath)
    transcription = result["text"]
    
    st.subheader("Transcribed Consultation")
    st.write(transcription)

    # Extract medical information using SciSpacy
    doc = nlp(transcription)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    st.subheader("Extracted Medical Information")
    for entity_text, entity_label in entities:
        st.write(f"**{entity_label}:** {entity_text}")

    # Summarize or structure report using LLM
    st.subheader("Generate Structured Medical Report")
    prompt = f"Generate a structured medical report based on this consultation transcription:\n{transcription}"
    response = llm([HumanMessage(content=prompt)])
    report = response.content

    st.text_area("Medical Report", value=report, height=300)

    # Download report options
    st.subheader("Download Report")

    # PDF Download
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, report)

    pdf_filename = f"medical_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_output = f"/tmp/{pdf_filename}"
    pdf.output(pdf_output)

    with open(pdf_output, "rb") as f:
        st.download_button("Download PDF", f, file_name=pdf_filename, mime="application/pdf")

    # Word Download
    docx = Document()
    docx.add_heading('Medical Report', 0)
    docx.add_paragraph(report)

    docx_filename = f"medical_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    docx_output = f"/tmp/{docx_filename}"
    docx.save(docx_output)

    with open(docx_output, "rb") as f:
        st.download_button("Download Word Document", f, file_name=docx_filename, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
