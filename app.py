from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os
import tempfile
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Test Paper Generator",
    page_icon="📝",
    layout="wide"
)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False


def load_documents(uploaded_files) -> List:
    """Load PDF documents from uploaded files"""
    all_docs = []
    
    for uploaded_file in uploaded_files:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        all_docs.extend(docs)
        
        # Clean up temp file
        os.unlink(tmp_path)
    
    return all_docs


def create_vectorstore(documents, api_key):
    """Create vector store from documents using HuggingFace embeddings (free, no quota limits)"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore


def generate_test_paper(vectorstore, api_key, subject, topic, num_mcq, num_short, num_long, difficulty):
    """Generate test paper using LangChain and Gemini"""
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        google_api_key=api_key,
        temperature=0.7
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are an expert CBSE Class 10 board exam paper setter. Using the following context from textbooks and previous year questions, 
    create a test paper for {subject} on the topic: {topic}.
    
    Context:
    {context}
    
    Requirements:
    - Difficulty Level: {difficulty}
    - {num_mcq} Multiple Choice Questions (1 mark each)
    - {num_short} Short Answer Questions (2-3 marks each)
    - {num_long} Long Answer Questions (5 marks each)
    
    Format the test paper professionally with:
    1. Paper heading with subject and topic
    2. Clear section divisions (Section A: MCQs, Section B: Short Answer, Section C: Long Answer)
    3. Proper numbering and mark allocation
    4. Questions that match CBSE board exam style
    5. For MCQs, provide 4 options (a, b, c, d)
    
    Generate the complete test paper now:
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    retrieval_chain = create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": 6}),
        document_chain
    )
    
    result = retrieval_chain.invoke({
        "input": f"Generate test paper for {subject} on topic {topic}",
        "subject": subject,
        "topic": topic,
        "num_mcq": num_mcq,
        "num_short": num_short,
        "num_long": num_long,
        "difficulty": difficulty
    })
    
    return result['answer']


# Main UI
st.title("📝 CBSE Class 10 Test Paper Generator")
st.markdown("Upload your textbook and PYQs to generate practice test papers!")

# Sidebar for API key and configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Get API key from environment or allow override
    default_api_key = os.getenv("GEMINI_API_KEY", "")
    
    if default_api_key:
        st.success("✅ API Key loaded from .env file")
        api_key = default_api_key
        if st.checkbox("Override API Key"):
            api_key = st.text_input(
                "Enter Different API Key", 
                type="password", 
                help="Override the .env API key"
            )
    else:
        st.warning("⚠️ No API key found in .env file")
        api_key = st.text_input(
            "Enter Gemini API Key", 
            type="password", 
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )
    
    st.markdown("---")
    st.markdown("### 📚 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload Textbook & PYQ PDFs",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload your textbook chapters and previous year question papers"
    )
    
    if uploaded_files and api_key:
        if st.button("Process Documents", type="primary"):
            with st.spinner("Processing documents... This may take a minute."):
                try:
                    documents = load_documents(uploaded_files)
                    st.session_state.vectorstore = create_vectorstore(documents, api_key)
                    st.session_state.documents_loaded = True
                    st.success(f"✅ Processed {len(documents)} pages successfully!")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")


# Main content area
if not api_key:
    st.warning("⚠️ Please enter your Gemini API key in the sidebar to continue")
elif not st.session_state.documents_loaded:
    st.info("📤 Upload your textbook and PYQ PDFs in the sidebar and click 'Process Documents' to get started")
else:
    st.success("✅ Documents loaded! Configure your test paper below:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        subject = st.text_input("Subject", placeholder="e.g., Computer Science")
        topic = st.text_input("Topic/Chapter", placeholder="e.g., Python Functions")
        difficulty = st.select_slider(
            "Difficulty Level",
            options=["Easy", "Medium", "Hard"],
            value="Medium"
        )
    
    with col2:
        num_mcq = st.number_input("Number of MCQs (1 mark each)", min_value=0, max_value=20, value=10)
        num_short = st.number_input("Short Answer Questions (2-3 marks)", min_value=0, max_value=10, value=5)
        num_long = st.number_input("Long Answer Questions (5 marks)", min_value=0, max_value=5, value=3)
    
    total_marks = (num_mcq * 1) + (num_short * 2.5) + (num_long * 5)
    st.info(f"📊 Total Marks: {total_marks}")
    
    if st.button("🎯 Generate Test Paper", type="primary", use_container_width=True):
        if not subject or not topic:
            st.error("Please enter both subject and topic")
        else:
            with st.spinner("🤖 AI is generating your test paper... Please wait..."):
                try:
                    test_paper = generate_test_paper(
                        st.session_state.vectorstore,
                        api_key,
                        subject,
                        topic,
                        num_mcq,
                        num_short,
                        num_long,
                        difficulty
                    )
                    
                    st.markdown("---")
                    st.markdown("## 📄 Generated Test Paper")
                    st.markdown(test_paper)
                    
                    st.download_button(
                        label="⬇️ Download Test Paper",
                        data=test_paper,
                        file_name=f"{subject}_{topic}_test_paper.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating test paper: {str(e)}")


# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with LangChain, Gemini AI & Streamlit | For CBSE Class 10 Students</p>
    </div>
""", unsafe_allow_html=True)
