import streamlit as st
import torch
import whisper
import spacy
import os
import datetime
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from fpdf import FPDF
from docx import Document
from pydub import AudioSegment  # ✅ NEW

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
            st.success("Patient registered!")

# ---------- Audio Upload ----------

st.subheader("Upload Audio Consultation")

audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        
        # Save uploaded audio properly
        audio = AudioSegment.from_file(audio_file)
        audio.export(tmp_path, format="wav")  # ✅ Always export as wav

    with st.spinner("Transcribing..."):
        result = whisper_model.transcribe(tmp_path)
        transcription = result["text"]

    st.subheader("Transcription")
    st.write(transcription)

    # ---------- NLP Processing ----------

    with st.spinner("Extracting key information..."):
        doc = nlp(transcription)

        symptoms = [ent.text for ent in doc.ents if ent.label_ == "SYMPTOM"]
        diseases = [ent.text for ent in doc.ents if ent.label_ == "DISEASE"]
        medications = [ent.text for ent in doc.ents if ent.label_ == "DRUG"]

    st.subheader("Extracted Information")
    st.write("**Symptoms:**", symptoms)
    st.write("**Diseases:**", diseases)
    st.write("**Medications:**", medications)

    # ---------- LLM Summary ----------

    with st.spinner("Generating summary..."):
        message = HumanMessage(content=f"Summarize the following patient consultation: {transcription}")
        summary = llm([message]).content

    st.subheader("Consultation Summary")
    st.write(summary)

    # ---------- Report Export ----------

    export_format = st.selectbox("Export Report As", ["PDF", "Word (.docx)"])

    if st.button("Download Report"):
        if export_format == "PDF":
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, summary)
            pdf_path = os.path.join(tempfile.gettempdir(), "report.pdf")
            pdf.output(pdf_path)
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, file_name="patient_report.pdf")
        else:
            docx = Document()
            docx.add_heading("Patient Consultation Summary", 0)
            docx.add_paragraph(summary)
            docx_path = os.path.join(tempfile.gettempdir(), "report.docx")
            docx.save(docx_path)
            with open(docx_path, "rb") as f:
                st.download_button("Download Word Document", f, file_name="patient_report.docx")
