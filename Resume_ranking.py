
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text.strip() if text else "No readable text found."

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes  # Combine job description with resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()

    return cosine_similarities

# Streamlit app
st.title("AI Resume Screening & Candidate Ranking System")

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description", key="job_desc")

# File uploader (Fixed duplicate issue with unique key)
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True, key="resume_upload")

if uploaded_files and job_description:
    st.subheader("Processing Resumes...")
    
    # Extract text from uploaded resumes
    resume_texts = [extract_text_from_pdf(file) for file in uploaded_files]
    resume_names = [file.name for file in uploaded_files]

    # Rank resumes
    similarity_scores = rank_resumes(job_description, resume_texts)

    # Create DataFrame for results (Fixed variable name issue)
    results_df = pd.DataFrame({
        "Resume Name": resume_names,
        "Score": similarity_scores
    }).sort_values(by="Score", ascending=False)

    # Display ranked resumes
    st.subheader("Ranked Resumes")
    st.write(results_df)

    # Plot results
    st.subheader("Ranking Visualization")
    fig, ax = plt.subplots()
    ax.barh(results_df["Resume Name"], results_df["Score"], color="skyblue")
    ax.set_xlabel("Similarity Score")
    ax.set_ylabel("Resume Name")
    ax.set_title("Resume Ranking Based on Job Description")
    ax.invert_yaxis()
    st.pyplot(fig)
