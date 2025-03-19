import streamlit as st
import re
import PyPDF2
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure Google Generative AI
API_KEY = "AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c"  # Replace with your actual API key
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to extract details using regex and Gemini API
def extract_details(resume_text):
    # Extract Name
    name = re.search(r"([A-Z][a-z]+ [A-Z][a-z]+)", resume_text)
    name = name.group(0) if name else "Unknown"

    # Extract Gender (assuming it's mentioned)
    gender = re.search(r"(Male|Female|Other)", resume_text, re.IGNORECASE)
    gender = gender.group(0) if gender else "Unknown"

    # Extract Date of Birth (assuming it's in DD/MM/YYYY or similar format)
    dob = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", resume_text)
    dob = dob.group(0) if dob else "Unknown"

    # Use Gemini API to extract skills, education, experience, projects, and research
    prompt = f"Extract the following details from the resume text:\n\nSkills:\nEducational Qualifications:\nWork Experience:\nProjects:\nResearch:\n\nResume Text:\n{resume_text}"
    response = model.generate_content(prompt)
    extracted_details = response.text

    return {
        "Name": name,
        "Gender": gender,
        "Date of Birth": dob,
        "Details": extracted_details
    }

# Function to calculate resume score (custom formula out of 100)
def calculate_score(details):
    score = 0

    # Score for Skills (max 30)
    skills = details["Details"].count("Skills:")
    score += min(skills * 5, 30)  # 5 points per skill, capped at 30

    # Score for Education (max 20)
    education = details["Details"].count("Educational Qualifications:")
    score += min(education * 4, 20)  # 4 points per qualification, capped at 20

    # Score for Work Experience (max 25)
    experience = details["Details"].count("Work Experience:")
    score += min(experience * 5, 25)  # 5 points per experience, capped at 25

    # Score for Projects (max 15)
    projects = details["Details"].count("Projects:")
    score += min(projects * 3, 15)  # 3 points per project, capped at 15

    # Score for Research (max 10)
    research = details["Details"].count("Research:")
    score += min(research * 2, 10)  # 2 points per research, capped at 10

    return score

# Function to generate feedback and suggestions using Gemini API
def generate_feedback_and_suggestions(resume_text, job_description):
    # Feedback for resume improvement
    feedback_prompt = f"Provide feedback to improve the following resume for the job description:\n\nResume:\n{resume_text}\n\nJob Description:\n{job_description}"
    feedback_response = model.generate_content(feedback_prompt)
    feedback = feedback_response.text

    # Job opportunities based on resume
    suggestions_prompt = f"Suggest job opportunities and roles suitable for the following resume:\n\nResume:\n{resume_text}"
    suggestions_response = model.generate_content(suggestions_prompt)
    suggestions = suggestions_response.text

    return feedback, suggestions

# Streamlit App
st.title("Resume Analyzer and Skill Enhancement Recommender")
st.write("Upload your resume in PDF format to analyze and get personalized feedback.")

# Upload PDF file
uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp_resume.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Step 1: Extract text from PDF
    resume_text = extract_text_from_pdf("temp_resume.pdf")

    # Step 2: Extract details from resume text
    details = extract_details(resume_text)

    # Display extracted details in a table
    st.subheader("Extracted Details")
    st.table(pd.DataFrame(list(details.items()), columns=["Field", "Value"]))

    # Step 3: Calculate resume score
    score = calculate_score(details)
    st.subheader(f"Resume Score: {score}/100")

    # Step 4: Display reason for the score
    st.subheader("Reason for Score")
    st.write(f"The score is calculated based on the following criteria:")
    st.write("- **Skills**: 5 points per skill (max 30)")
    st.write("- **Education**: 4 points per qualification (max 20)")
    st.write("- **Work Experience**: 5 points per experience (max 25)")
    st.write("- **Projects**: 3 points per project (max 15)")
    st.write("- **Research**: 2 points per research (max 10)")

    # Step 5: Generate feedback and suggestions
    job_description = "Looking for a software engineer with expertise in Python, Java, and machine learning."
    feedback, suggestions = generate_feedback_and_suggestions(resume_text, job_description)

    st.subheader("Feedback for Resume Improvement")
    st.write(feedback)

    st.subheader("Suggested Job Opportunities")
    st.write(suggestions)
