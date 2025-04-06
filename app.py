# Write the Streamlit app to app.py
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

@st.cache_resource
def load_nlp_model():
    return spacy.load("en_core_sci_md")

@st.cache_resource
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("base", device=device)

@st.cache_resource
def load_llm_model():
    os.environ["GOOGLE_API_KEY"] = "AIzaSyAbweJIGcw6SRu1MEmWW7V0jjnCdFiqExE"
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Load models once
nlp = load_nlp_model()
model = load_whisper_model()
llm = load_llm_model()


# Persist patient database across interactions
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
        sex = st.radio("Sex", ("Male", "Female", "Other"))
        if st.button("Register Patient"):
            new_pid = str(len(patient_db) + 1)
            patient_db[new_pid] = {"name": name, "age": age, "sex": sex, "history": []}
            st.success(f"Patient registered with ID: {new_pid}")
            st.session_state.patient_db = patient_db
            pid = new_pid
else:
    st.info("Enter a Patient ID to search for an existing record or leave blank to register a new patient.")

uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a", "flac"])

if uploaded_file and pid:
    with NamedTemporaryFile(delete=False) as temp_audio:
        temp_audio.write(uploaded_file.read())
        audio_path = temp_audio.name

    st.info("Processing audio file... ⏳")
    result = model.transcribe(audio_path)
    transcription = result["text"]
    st.success("Transcription Complete! ✅")
    st.text_area("📝 Transcribed Text:", transcription, height=150)

    doc = nlp(transcription)
    medical_conditions = [ent.text for ent in doc.ents]

    st.subheader("🔬 Extracted Medical Conditions")
    st.write(medical_conditions if medical_conditions else "No conditions detected.")

    if st.button("Generate Medical Report"):
        consultation_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt = f"""
        Convert the following patient consultation into a structured medical report.
        **Patient Name:** {patient_db[pid]['name']}
        **Age/Sex:** {patient_db[pid]['age']}/{patient_db[pid]['sex']}
        **Date of Consultation:** {consultation_date}
        **Transcribed Conversation:** {transcription}
        **Extracted Medical Conditions:** {', '.join(medical_conditions)}
        """
        response = llm([HumanMessage(content=prompt)])
        structured_report = response.content

        patient_db[pid]['history'].append({
            "date": consultation_date,
            "transcription": transcription,
            "medical_conditions": medical_conditions,
            "report": structured_report
        })
        st.session_state.patient_db = patient_db

        st.subheader("📄 Generated Medical Report")
        st.write(structured_report)

        report_content = "".join(
            f"\n\nDate: {entry['date']}\n{entry['report']}" for entry in patient_db[pid]['history']
        )

        def generate_pdf(content, filename):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, content)
            pdf.output(filename)

        def generate_docx(content, filename):
            doc = Document()
            doc.add_paragraph(content)
            doc.save(filename)

        pdf_filename = f"Medical_Report_{pid}.pdf"
        generate_pdf(report_content, pdf_filename)
        with open(pdf_filename, "rb") as pdf_file:
            st.download_button("📥 Download PDF Report", pdf_file, pdf_filename, "application/pdf")

        docx_filename = f"Medical_Report_{pid}.docx"
        generate_docx(report_content, docx_filename)
        with open(docx_filename, "rb") as docx_file:
            st.download_button("📥 Download DOCX Report", docx_file, docx_filename, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

        st.success("Report saved and updated! ✅")