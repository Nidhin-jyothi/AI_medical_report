import streamlit as st
import requests
import spacy
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from fpdf import FPDF
from docx import Document

# Load SciSpaCy Model
nlp = spacy.load("en_core_sci_md")

# Set up API keys
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]

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
audio_file = st.file_uploader("Upload an audio file (.mp3, .wav)", type=["mp3", "wav"])

if audio_file:
    with st.spinner('Transcribing with Whisper...'):
        # Send audio to Replicate's Whisper model
        url = "https://api.replicate.com/v1/predictions"
        headers = {
            "Authorization": f"Token {os.environ['REPLICATE_API_TOKEN']}",
            "Content-Type": "application/json"
        }
        files = {"file": audio_file}
        upload_response = requests.post(
            "https://dreambooth-api-experimental.replicate.com/v1/files",
            headers=headers,
            files={"file": (audio_file.name, audio_file, audio_file.type)}
        )
        file_url = upload_response.json()["url"]

        body = {
            "version": "f481705f0d682c9b2a3a99d3c9c2e46ee87d764b8713b8a6de998d0df5c04e4a",  # whisper-1
            "input": {"audio": file_url}
        }
        response = requests.post(url, headers=headers, json=body)
        prediction = response.json()

        transcription = prediction["prediction"]

        st.success("Transcription Complete!")
        st.text_area("Transcribed Text", transcription, height=200)

        # Medical entity extraction
        doc = nlp(transcription)
        medical_terms = [ent.text for ent in doc.ents]

        st.subheader("Extracted Medical Terms")
        st.write(medical_terms)

        # Summarization / Report generation using Gemini
        user_prompt = f"Generate a detailed medical report based on this consultation: {transcription}"
        response = llm([HumanMessage(content=user_prompt)])
        final_report = response.content

        st.subheader("Generated Medical Report")
        st.write(final_report)

        if st.button("Download Report as PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, final_report)
            pdf_output = "medical_report.pdf"
            pdf.output(pdf_output)
            with open(pdf_output, "rb") as f:
                st.download_button("Download PDF", f, file_name="medical_report.pdf")

        if st.button("Download Report as Word"):
            docx = Document()
            docx.add_heading("Medical Report", 0)
            docx.add_paragraph(final_report)
            docx_output = "medical_report.docx"
            docx.save(docx_output)
            with open(docx_output, "rb") as f:
                st.download_button("Download DOCX", f, file_name="medical_report.docx")
