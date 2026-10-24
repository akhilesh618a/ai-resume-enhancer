import streamlit as st
import pandas as pd
import re
import nltk
import pdfplumber
import docx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings
import os

from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

nltk.download("stopwords")

# ---------------------------
# DATA FILES
# ---------------------------

USERS_FILE = "users.csv"
JOBS_FILE = "jobs_posted.csv"
APPLY_FILE = "applications.csv"

# ---------------------------
# INIT FILES WITH HEADERS
# ---------------------------

if not os.path.exists(USERS_FILE):
    pd.DataFrame(columns=["name","email","password","role"]).to_csv(USERS_FILE,index=False)

if not os.path.exists(JOBS_FILE):
    pd.DataFrame(columns=["title","description"]).to_csv(JOBS_FILE,index=False)

if not os.path.exists(APPLY_FILE):
    pd.DataFrame(columns=["candidate","job"]).to_csv(APPLY_FILE,index=False)

# ---------------------------
# STOPWORDS
# ---------------------------

stop_words = set(stopwords.words("english"))

# ---------------------------
# MODEL LOAD
# ---------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------------------
# TEXT CLEANING
# ---------------------------

def clean_text(text):

    text = str(text).lower()
    text = re.sub(r'[^a-z ]','',text)

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)

# ---------------------------
# RESUME READER
# ---------------------------

def read_resume(file):

    text = ""

    if file.name.endswith(".pdf"):

        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t

    elif file.name.endswith(".docx"):

        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"

    elif file.name.endswith(".txt"):

        text = file.read().decode()

    elif file.name.endswith(".csv"):

        df = pd.read_csv(file)
        text = " ".join(df.astype(str).values.flatten())

    return text

# ---------------------------
# GAUGE METER
# ---------------------------

def gauge(score):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix':"/10"},
        gauge={'axis':{'range':[0,10]}}
    ))

    st.plotly_chart(fig,use_container_width=True)

# ---------------------------
# LOGIN SYSTEM
# ---------------------------

st.title("AI Resume Analyzer & Job Portal")

menu = ["Login","Register"]
choice = st.sidebar.selectbox("Menu",menu)

users = pd.read_csv(USERS_FILE)

# ---------------------------
# REGISTER
# ---------------------------

if choice == "Register":

    st.subheader("Create Account")

    name = st.text_input("Name")
    email = st.text_input("Email")
    password = st.text_input("Password",type="password")
    role = st.selectbox("Role",["Candidate","Recruiter"])

    if st.button("Register"):

        if email in users.get("email",[]).values:

            st.error("User already exists")

        else:

            new = pd.DataFrame([[name,email,password,role]],
                               columns=["name","email","password","role"])

            users = pd.concat([users,new])
            users.to_csv(USERS_FILE,index=False)

            st.success("Account created")

# ---------------------------
# LOGIN
# ---------------------------

if choice == "Login":

    email = st.text_input("Email")
    password = st.text_input("Password",type="password")

    if st.button("Login"):

        user = users[(users.email==email) & (users.password==password)]

        if user.empty:

            st.error("Invalid login")

        else:

            st.session_state.role = user.iloc[0]["role"]
            st.session_state.user = email

            st.success("Logged in")

# ---------------------------
# MAIN PORTAL
# ---------------------------

if "role" in st.session_state:

    role = st.session_state.role

    nav = st.sidebar.selectbox(
        "Navigation",
        ["Home","Candidate Portal","Recruiter Portal","Logout"]
    )

    # -------------------
    # HOME
    # -------------------

    if nav == "Home":

        st.header("Welcome to AI Resume Analyzer")
        st.write("Analyze resumes, match with jobs and improve ATS score.")

    # -------------------
    # CANDIDATE PORTAL
    # -------------------

    if nav == "Candidate Portal":

        if role != "Candidate":
            st.warning("Only candidates allowed")

        else:

            st.header("Candidate Portal")

            resume = st.file_uploader(
                "Upload Resume",
                type=["pdf","docx","txt","csv"]
            )

            jd_input = st.text_area("Paste Job Description (optional)")

            if st.button("Analyze Resume"):

                if resume is None:

                    st.error("Upload resume first")

                else:

                    resume_text = read_resume(resume)
                    resume_lower = resume_text.lower()

                    word_count = len(resume_text.split())

                    st.metric("Resume Words",word_count)

                    sections = ["education","experience","skills","projects","certifications"]

                    ats_score = sum([20 for s in sections if s in resume_lower])

                    if 400 < word_count < 900:
                        ats_score += 10

                    st.metric("ATS Score",ats_score)

                    resume_clean = clean_text(resume_text)
                    resume_emb = model.encode([resume_clean])

                    if jd_input != "":

                        jd_clean = clean_text(jd_input)
                        jd_emb = model.encode([jd_clean])

                        score = cosine_similarity(resume_emb,jd_emb)[0][0]

                        rating = score*10

                        st.subheader("Resume Rating")
                        gauge(rating)

                        st.metric("Match %",round(score*100,2))

            # ------------------
            # SHOW JOBS
            # ------------------

            jobs = pd.read_csv(JOBS_FILE)

            if not jobs.empty:

                st.subheader("Available Jobs")

                for i,row in jobs.iterrows():

                    st.write("###",row["title"])
                    st.write(row["description"])

                    if st.button("Apply",key=f"apply_{i}"):

                        apps = pd.read_csv(APPLY_FILE)

                        new = pd.DataFrame([[st.session_state.user,row["title"]]],
                                           columns=["candidate","job"])

                        apps = pd.concat([apps,new])
                        apps.to_csv(APPLY_FILE,index=False)

                        st.success("Applied successfully")

    # -------------------
    # RECRUITER PORTAL
    # -------------------

    if nav == "Recruiter Portal":

        if role != "Recruiter":

            st.warning("Only recruiters allowed")

        else:

            st.header("Recruiter Portal")

            title = st.text_input("Job Title")
            jd = st.text_area("Job Description")

            if st.button("Post Job"):

                if title == "" or jd == "":

                    st.warning("Enter job title and description")

                else:

                    jobs = pd.read_csv(JOBS_FILE)

                    new = pd.DataFrame([[title,jd]],
                                       columns=["title","description"])

                    jobs = pd.concat([jobs,new])
                    jobs.to_csv(JOBS_FILE,index=False)

                    st.success("Job Posted")

            st.subheader("Applicants")

            apps = pd.read_csv(APPLY_FILE)

            if not apps.empty:

                st.dataframe(apps)

            else:

                st.info("No applicants yet")

    # -------------------
    # LOGOUT
    # -------------------

    if nav == "Logout":

        st.session_state.clear()
        st.success("Logged out")

# ---------------------------
# FOOTER
# ---------------------------

st.markdown("---")
st.write("© 2026 AI Resume Analyzer")