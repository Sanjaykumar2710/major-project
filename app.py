# app.py
import streamlit as st
import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import pandas as pd
from typing import List
from google.generativeai import GenerativeModel
from sklearn.feature_extraction.text import CountVectorizer
import tempfile
import shutil
import pdfplumber
import google.generativeai as genai

genai.configure(api_key="AIzaSyC6aIEuesKICOEEtVBadDeqi0-hqNmmm9c")

# ========================
# Setup and Constants
# ========================
RESUME_DIR = "data/resumes"
JD_FILE = "data/job_description.txt"
os.makedirs(RESUME_DIR, exist_ok=True)

st.set_page_config(page_title="🧠 Intelligent Hiring Assistant", layout="wide")
st.title(":factory: Intelligent Hiring Assistant for Industrial Roles")

# ========================
# Upload Resumes
# ========================
st.sidebar.header(":card_file_box: Upload Resumes")
uploaded_files = st.sidebar.file_uploader("Upload resumes (PDF only)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(RESUME_DIR, file.name), "wb") as f:
            f.write(file.read())
    st.sidebar.success(":white_check_mark: Resumes uploaded successfully!")

# ========================
# Load Job Description
# ========================
st.sidebar.header(":page_facing_up: Job Description")
if os.path.exists(JD_FILE):
    with open(JD_FILE, "r") as f:
        job_description = f.read()
else:
    job_description = ""

job_description = st.sidebar.text_area("Paste job description here:", job_description, height=200)

# ========================
# Agents
# ========================
class ResumeParsingAgent:
    def parse_resumes(self, folder):
        parsed = []
        for file in os.listdir(folder):
            if file.endswith(".pdf"):
                try:
                    with pdfplumber.open(os.path.join(folder, file)) as pdf:
                        text = "\n".join(page.extract_text() or '' for page in pdf.pages)
                    parsed.append({"filename": file, "text": text})
                except Exception as e:
                    st.error(f"Error parsing {file}: {e}")
        return parsed

class SkillExtractionAgent:
    def extract_keywords(self, text: str, top_k: int = 15):
        vectorizer = CountVectorizer(stop_words='english', max_features=top_k)
        X = vectorizer.fit_transform([text])
        return vectorizer.get_feature_names_out().tolist()

    def compute_gap(self, resume_text, jd_text):
        resume_keywords = self.extract_keywords(resume_text)
        jd_keywords = self.extract_keywords(jd_text)
        missing = list(set(jd_keywords) - set(resume_keywords))
        return ", ".join(missing) if missing else "No major gaps"

class MatchingAgent:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def build_index(self, texts):
        embeddings = self.model.encode(texts)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index, embeddings

    def search(self, index, resumes, job_description, top_k=5):
        job_vec = self.model.encode([job_description])
        D, I = index.search(job_vec, top_k)
        scores = {resumes[i]['filename']: 1 / (1 + D[0][k]) for k, i in enumerate(I[0])}
        return scores

class GeminiQuestionGenerator:
    def __init__(self):
        self.model = GenerativeModel("gemini-1.5-flash")

    def generate_questions(self, jd: str) -> List[str]:
        prompt = f"""
        Generate 5 technical interview questions tailored to the following job description:

        {jd}
        
        Return the list as plain text.
        """
        response = self.model.generate_content(prompt)
        return response.text.strip().split("\n")

class ShortlistingAgent:
    def rank(self, resumes, scores, gaps):
        result = []
        for r in resumes:
            name = r["filename"]
            result.append({
                "Candidate": name,
                "Fit Score": round(scores.get(name, 0), 2),
                "Skill Gap": gaps.get(name, "N/A"),
                "Expected Retention": "12+ months",
                "Onboarding Time": "2 weeks"
            })
        return pd.DataFrame(result).sort_values(by="Fit Score", ascending=False)

# ========================
# Run Pipeline
# ========================
if st.button(":mag: Run Hiring Assistant"):
    if not job_description.strip():
        st.error("Please paste a job description.")
    elif not os.listdir(RESUME_DIR):
        st.error("Please upload at least one resume.")
    else:
        with st.spinner(":gear: Processing resumes..."):
            parser = ResumeParsingAgent()
            matcher = MatchingAgent()
            skill_agent = SkillExtractionAgent()
            question_agent = GeminiQuestionGenerator()
            shortlist_agent = ShortlistingAgent()

            resumes = parser.parse_resumes(RESUME_DIR)
            texts = [r["text"] for r in resumes]
            index, _ = matcher.build_index(texts)
            scores = matcher.search(index, resumes, job_description)
            gaps = {r["filename"]: skill_agent.compute_gap(r["text"], job_description) for r in resumes}
            ranked_df = shortlist_agent.rank(resumes, scores, gaps)
            questions = question_agent.generate_questions(job_description)

        st.subheader(":bar_chart: Ranked Candidates")
        st.dataframe(ranked_df, use_container_width=True)

        st.subheader("🧠 Auto-Generated Interview Questions")
        for q in questions:
            st.markdown(f"- {q}")

        st.subheader(":triangular_ruler: Skill Gap Analysis")
        st.dataframe(pd.DataFrame.from_dict(gaps, orient="index", columns=["Gap Summary"]))
