import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pytesseract
from PIL import Image

# Configure Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Ensure this path is correct

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function for extracting text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function for extracting text from images
def get_image_text(image_files):
    text = ""
    for image_file in image_files:
        image = Image.open(image_file)
        text += pytesseract.image_to_string(image)
    return text

# Function for creating chunks from extracted text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function for creating vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Template for chatbot prompt
template = """
Role: You are a Medical Report Analyzer assisting a human user.

Task: Analyze the extracted part of a PDF or image, which is a medical test report of any type (e.g., Blood Test, Urine Test, Spit Test, UGI Endoscopy, ECG, EEG, Biopsy, Allergy Test, Stool Test, Genetic Test, Bone Density Test, Lipid Panel, Liver Function Test, Kidney Function Test, Thyroid Function Test).

Objective:
- Identify and explain any symptoms of diseases found in the report.
- Provide relevant precautions or recommendations based on the identified symptoms.

Instructions:
- If you do not understand the context or the content of the report, clearly state that you do not have enough information to provide an accurate analysis.
- Do not provide incorrect or misleading information.

context: \n{context}?\n

question:\n{question}\n
Chatbot:
"""

# Function for creating conversation chain
def get_conversation_chain():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
    prompt = PromptTemplate(input_variables=['question', "context"], template=template)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function for loading FAISS index
def load_faiss_index(pickle_file):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_index = FAISS.load_local(pickle_file, embeddings=embeddings, allow_dangerous_deserialization=True)
    return faiss_index

# Function for handling user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = load_faiss_index("faiss_index")
    docs = new_db.similarity_search(user_question)

    chain = get_conversation_chain()
    response = chain({"input_documents": docs, "question": user_question})
    st.write("**Answer:**", response['output_text'])

# Streamlit app configuration
st.set_page_config(
    page_title="Medical Report Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="auto",
)

with st.sidebar:
    st.title("Upload Medical Reports")
    pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=['pdf'])
    image_files = st.file_uploader("Upload Image Files", accept_multiple_files=True, type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'])

    if st.button("Submit"):
        with st.spinner("Uploading..."):
            raw_text = ""
            if pdf_docs:
                raw_text += get_pdf_text(pdf_docs)
            if image_files:
                raw_text += get_image_text(image_files)

            if raw_text:
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.success("VectorDB Upload Finished")
            else:
                st.error("No text extracted from the provided files.")

def main():
    st.header("Chat with Medical Reports")
    user_question = st.text_input("Ask your questions...")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
