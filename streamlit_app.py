import streamlit as st
import requests
import spacy
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from fpdf import FPDF
from docx import Document

# Set up API keys at the top
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]

# Load SpaCy model (cached to avoid reloading)
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_sci_md")

nlp = load_spacy_model()

# Set up LLM (Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Patient database persistence
if "patient_db" not in st.session_state:
    st.session_state.patient_db = {}

patient_db = st.session_state.patient_db

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
        sex = st.radio("Sex", ["Male", "Female", "Other"])
        if st.button("Register Patient"):
            patient_db[pid] = {"name": name, "age": age, "sex": sex}
            st.success(f"Patient {pid} registered!")

st.subheader("Upload Consultation Audio")
audio_file = st.file_uploader("Upload an audio file (.wav, .mp3, etc.)", type=["wav", "mp3", "m4a"])

if audio_file:
    # Send to Replicate's Whisper model for transcription
    with st.spinner("Transcribing audio..."):
        audio_bytes = audio_file.read()

        # Send request to Replicate API (you must have set REPLICATE_API_TOKEN)
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers={
                "Authorization": f"Token {os.environ['REPLICATE_API_TOKEN']}",
                "Content-Type": "application/json",
            },
            json={
                "version": "your-whisper-model-version-id-here",  # You MUST put correct Whisper model version ID
                "input": {"audio": audio_bytes}
            }
        )
        prediction = response.json()
        if 'error' in prediction:
            st.error(f"Error from Replicate API: {prediction['error']}")
        else:
            transcription = prediction["prediction"]["text"]
            st.subheader("Transcription")
            st.write(transcription)

            # Extract information using SciSpaCy
            doc = nlp(transcription)
            medical_entities = [ent.text for ent in doc.ents if ent.label_ in ["DISEASE", "SYMPTOM", "TREATMENT"]]

            st.subheader("Extracted Medical Information")
            st.write(medical_entities)

            # Generate structured report using LLM
            with st.spinner("Generating Medical Report..."):
                prompt = f"""You are a medical assistant. Create a structured medical report based on this consultation transcript:

{transcription}

Include sections like Chief Complaint, History of Present Illness, Assessment, and Plan."""
                report = llm([HumanMessage(content=prompt)]).content

                st.subheader("Generated Medical Report")
                st.write(report)

                # Option to download report as PDF or DOCX
                if st.button("Download Report as PDF"):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    for line in report.split("\n"):
                        pdf.multi_cell(0, 10, line)
                    pdf_output = "medical_report.pdf"
                    pdf.output(pdf_output)
                    with open(pdf_output, "rb") as file:
                        st.download_button("Download PDF", data=file, file_name="medical_report.pdf")

                if st.button("Download Report as DOCX"):
                    doc = Document()
                    doc.add_heading("Medical Report", 0)
                    for para in report.split("\n"):
                        doc.add_paragraph(para)
                    docx_output = "medical_report.docx"
                    doc.save(docx_output)
                    with open(docx_output, "rb") as file:
                        st.download_button("Download DOCX", data=file, file_name="medical_report.docx")
