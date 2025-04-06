# AI-Powered Medical Documentation System

## Overview
This project is an AI-driven medical documentation system that transcribes doctor-patient consultations from audio files and generates structured medical reports. It leverages **Whisper** for transcription, **SciSpaCy** for extracting medical entities, and **Gemini (Google Generative AI)** for generating structured reports.

## Features
- **Speech-to-Text**: Converts doctor-patient consultations into text using Whisper.
- **Medical Condition Extraction**: Uses SciSpaCy to extract relevant medical conditions.
- **AI-Generated Reports**: Converts transcriptions into structured reports using Gemini AI.
- **Patient History Management**: Stores consultation history for each patient.
- **Downloadable Reports**: Exports medical reports as **PDF** or **DOCX**.

## Requirements
- Python 3.11+
- PyTorch
- OpenAI Whisper
- SciSpaCy (`en_core_sci_md` model)
- LangChain & Google Generative AI
- fpdf (for PDF export)
- python-docx (for DOCX export)

Install dependencies using:
```bash
pip install torch whisper spacy langchain-google-genai fpdf python-docx
python -m spacy download en_core_sci_md
```

## Usage
Run the Jupyter Notebook (`Medical_Documentation.ipynb`) and follow the interactive prompts to:
1. Upload an audio file.
2. Generate transcriptions & extract medical conditions.
3. Generate AI-powered structured reports.
4. Download reports in **PDF/DOCX** format.

## Future Enhancements
- Improve AI-generated reports with fine-tuned medical models.
- Integrate an EHR system for seamless medical data management.
- Enable real-time speech-to-text analysis.

---


