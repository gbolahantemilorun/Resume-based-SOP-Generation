from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from fpdf import FPDF
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import openai

# Load the env variables
load_dotenv(dotenv_path=".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create a title and header
st.title("SOP Generation App")
st.header("Generate SOP with your CV")

# User input
cv_file = st.file_uploader("Upload your CV (PDF)", type="pdf")
field_of_study = st.text_input("Field of Study for Postgraduate Study")
university_name = st.text_input("Name of the University")

# Check if the uploaded file is not none
if cv_file is not None:
    pdf_reader = PdfReader(cv_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    # Split the text into smaller chunks
    chunks = text_splitter.split_text(text)
    
    # Create an embedding
    embeddings = OpenAIEmbeddings()
    
    # Create a knowledge base
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    query = "Extract skills, education, experience and projects"
    # Search the knowledge base for documents related to the user's query
    docs = knowledge_base.similarity_search(query)
    
    # Initialize an OpenAI Model
    llm = OpenAI()
    
    # Load a question-answering chain using the OpenAI model
    chain = load_qa_chain(llm, chain_type="stuff")
    
    # Run the chain with the input document and user's question to generate a response
    cv_content = chain.run(input_documents=docs, question=query)
    
    # Generate SOP
    if st.button("Generate SOP"):

        # Generate SOP using GPT-3
        prompt = f"Generate a Statement of Purpose for postgraduate study in {field_of_study} at {university_name}. CV Content: {cv_content}"

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=500,
            temperature=0.7,
            n=1,
            stop=None
        )
        
        response_text = response['choices'][0]['text']

        # Display generated SOP
        st.subheader("Generated Statement of Purpose:")
        st.success(response_text)
        
        # Download button for PDF
        if st.button("Download as PDF"):
            # Save the generated SOP to a PDF file
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, response_text)
            pdf_filename = "downloaded_sop.pdf"
            pdf.output(pdf_filename)
            st.success("PDF Downloaded!")
        
    else:
        st.warning("Please upload your CV before generating the SOP.")