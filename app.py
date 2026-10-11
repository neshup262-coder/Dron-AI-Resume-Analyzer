import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download NLTK resources
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

# 1. Page Setup & Configuration
st.set_page_config(
    page_title="Resume Intelligence AI Pro", 
    page_icon="✨", 
    layout="wide"
)

# 2. Next-Level Ultra-Modern CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Global Font & Rich Darkish Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #0f172a 50%, #171033 100%);
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #f3f4f6;
        background-attachment: fixed;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Futuristic Header Banner */
    .hero-container {
        padding: 30px 0 15px 0;
        text-align: left;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1.5px;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #94a3b8;
        font-weight: 400;
        max-width: 750px;
        line-height: 1.6;
    }

    /* Ultra-Sleek Glassmorphism Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
        margin-bottom: 24px;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(129, 140, 248, 0.4);
        box-shadow: 0 20px 40px rgba(129, 140, 248, 0.15);
    }

    /* Next-Level Keyword Pill Badges */
    .keyword-pill {
        display: inline-block;
        background: rgba(129, 140, 248, 0.12);
        color: #a5b4fc;
        padding: 6px 14px;
        margin: 5px;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid rgba(129, 140, 248, 0.3);
        transition: all 0.2s ease;
    }
    .keyword-pill:hover {
        background: rgba(129, 140, 248, 0.3);
        color: #ffffff;
        transform: translateY(-2px);
    }

    /* Customizing Streamlit Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        border: none;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        transform: translateY(-1px);
    }

    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #818cf8;
    }
</style>
""", unsafe_allow_html=True)

# Highly Attractive & Polished Sidebar Control Panel
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 10px 0;'>
            <h2 style='color: #818cf8; font-weight: 800; letter-spacing: -0.5px;'>🧭 Control Panel</h2>
            <p style='color: #94a3b8; font-size: 0.85rem;'>AI-Powered Resume Optimizer</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        <div style='background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(129, 140, 248, 0.2); padding: 16px; border-radius: 14px; margin-bottom: 15px;'>
            <p style='color: #818cf8; font-weight: 700; margin-bottom: 8px;'>💡 Master Guidelines</p>
            <ul style='color: #94a3b8; font-size: 0.85rem; padding-left: 18px; margin: 0; line-height: 1.5;'>
                <li>Upload clean PDF resumes with selectable layers.</li>
                <li>Paste detailed Job Descriptions for deep semantic mapping.</li>
                <li>Analyze gaps and utilize missing keyword tags.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(192, 132, 252, 0.2); padding: 16px; border-radius: 14px;'>
            <p style='color: #c084fc; font-weight: 700; margin-bottom: 6px;'>⚡ Advanced Stack</p>
            <p style='color: #94a3b8; font-size: 0.83rem; margin: 0; line-height: 1.4;'>TF-IDF Vectors, Cosine Similarity, and NLTK NLP Tagging Engine.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.75rem;'>Engineered for Elite Professionals 🔥</p>", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Resume Intelligence Studio</div>
    <div class="hero-subtitle">Evaluate, audit, and upgrade your resume against top-tier tech job descriptions instantly using deep vector matching algorithms.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Helper Functions
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""
    
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return " ".join([word for word in words if word not in stop_words])

def calculate_similarity(resume_text, job_description):
    resume_processed = remove_stopwords(clean_text(resume_text))
    job_processed = remove_stopwords(clean_text(job_description))
    
    if not resume_processed or not job_processed:
        return 0.0, resume_processed, job_processed

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_processed, job_processed])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    return round(score, 2), resume_processed, job_processed

def extract_keywords(text, num_keywords=18):
    words = word_tokenize(text)
    words = [w for w in words if len(w) > 2]
    tagged_words = pos_tag(words)
    nouns = [w for w, pos in tagged_words if pos.startswith('NN') or pos.startswith('JJ')]
    word_freq = Counter(nouns)
    return [word for word, freq in word_freq.most_common(num_keywords)]

# Modern Two-Column Input Layout
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### **📄 Step 1: Upload Candidate Resume**")
    uploaded_file = st.file_uploader("Upload PDF file", type=['pdf'], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### **📋 Step 2: Paste Target Job Description**")
    job_description = st.text_area("Paste job description here", height=130, label_visibility="collapsed", placeholder="Paste required technical stacks, qualifications, and core duties...")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Centered Action Button Container
b_col1, b_col2, b_col3 = st.columns([1, 1.5, 1])
with b_col2:
    analyze_button = st.button("🚀 Execute Semantic Match Engine")

if analyze_button:
    if not uploaded_file:
        st.warning("⚠️ Please upload your resume PDF to begin semantic evaluation.")
    elif not job_description:
        st.warning("⚠️ Please provide the target job description text.")
    else:
        with st.spinner("✨ Running deep vector cosine matrix computations..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            if not resume_text:
                st.error("Could not parse text layers from this PDF. Try a different file.")
            else:
                similarity_score, resume_processed, job_processed = calculate_similarity(resume_text, job_description)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📊 Performance & Analytics Dashboard")

                # Results Layout Cards
                res_c1, res_c2 = st.columns([1, 1.8], gap="large")

                with res_c1:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.metric(label="Match Confidence Index", value=f"{similarity_score:.2f}%")
                    
                    if similarity_score < 40:
                        st.error("Critical Gap: Low Alignment ❌")
                    elif similarity_score < 70:
                        st.warning("Moderate Alignment: Needs Tuning ⚠️")
                    else:
                        st.success("High Performance: Strong Match 🔥")
                    st.markdown('</div>', unsafe_allow_html=True)

                with res_c2:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("<p style='font-weight: 600; margin-bottom: 12px; color: #94a3b8;'>Alignment Spectrum Bar</p>", unsafe_allow_html=True)
                    
                    # Dark-Theme Matplotlib Chart seamlessly blended with UI
                    fig, ax = plt.subplots(figsize=(6, 1.3))
                    fig.patch.set_facecolor('#0f172a')
                    ax.set_facecolor('#0f172a')
                    
                    colors = ['#f43f5e', '#fbbf24', '#10b981']
                    color_index = min(int(similarity_score // 34), 2)
                    
                    ax.barh([0], [similarity_score], color=colors[color_index], height=0.4, zorder=3)
                    ax.barh([0], [100], color='#1e293b', height=0.4, zorder=2)
                    
                    ax.set_xlim(0, 100)
                    ax.set_xticks([0, 25, 50, 75, 100])
                    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], color='#94a3b8', fontsize=9)
                    ax.set_yticks([])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_color('#334155')
                    ax.tick_params(axis='x', colors='#94a3b8')
                    
                    st.pyplot(fig)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Extracted Resume Text Preview Expander
                with st.expander("🔍 View Extracted Resume Text (PDF Parser Preview)"):
                    st.text_area("Parsed Text Content", resume_text, height=180)

                # Strategic Recommendations, Missing Keywords & Report Download
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 🧠 AI Strategic Gap Analysis & Optimization")
                
                job_keywords = set(extract_keywords(job_description, num_keywords=20))
                resume_words = set(resume_processed.split())
                missing_keywords = list(job_keywords - resume_words)

                if missing_keywords:
                    st.write("To maximize your selection probability, integrate these high-priority missing keywords directly into your summary or skills sections:")
                    
                    badges_html = "".join([f'<span class="keyword-pill">{kw}</span>' for kw in missing_keywords])
                    st.markdown(f'<div style="padding: 10px 0;">{badges_html}</div>', unsafe_allow_html=True)
                    
                    missing_str = ", ".join(missing_keywords)
                    
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        report_content = f"RESUME OPTIMIZATION REPORT\n" + "="*30 + f"\nMatch Score: {similarity_score}%\n\nMissing Keywords:\n{missing_str}\n"
                        st.download_button(
                            label="📥 Download Audit Report (.txt)",
                            data=report_content,
                            file_name="resume_optimization_report.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                else:
                    st.success("🎉 Outstanding! Your resume text covers all primary semantic requirements from the description.")
                
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    pass