import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

st.set_page_config(page_title="DocuTalks", layout="wide")

st.sidebar.title("⚙️ Configurations:")
st.sidebar.divider()
st.title("📄 DocuTalks")
st.markdown("##### Seamless Chat with Your PDF Documents!")
st.divider()

# This is the first API key input; no need to repeat it in the main function.
api_key = st.sidebar.text_input("Enter your Google API Key and Press Enter:", type="password", key="api_key_input")
st.sidebar.write("Google AI Studio: https://aistudio.google.com/app/apikey")
st.sidebar.write("Here's how to obtain it: https://sakhujayashofficia.wixsite.com/yashsakhuja/post/ai-applications-with-langchain-and-gemini")
st.sidebar.divider()

def get_pdf_text_with_metadata(pdf_docs):
    # This will hold tuples of (document_name, page_number, page_text)
    all_text = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text()
            if text:
                all_text.append({
                    "text": text,
                    "doc_name": pdf.name,  # Get the PDF file name
                    "page_num": page_num
                })
    return all_text

def get_text_chunks_with_metadata(pdf_metadata):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks_with_metadata = []

    for entry in pdf_metadata:
        chunks = text_splitter.split_text(entry["text"])
        for chunk in chunks:
            chunks_with_metadata.append({
                "chunk_text": chunk,
                "doc_name": entry["doc_name"],
                "page_num": entry["page_num"]
            })

    return chunks_with_metadata

def get_vector_store(text_chunks, api_key):
    # Extract chunk texts and associate them with their metadata
    chunk_texts = [chunk["chunk_text"] for chunk in text_chunks]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    # Store the metadata for each chunk as well
    vector_store = FAISS.from_texts(chunk_texts, embedding=embeddings, metadatas=[{
        "doc_name": chunk["doc_name"],
        "page_num": chunk["page_num"]
    } for chunk in text_chunks])

    try:
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error saving FAISS index: {str(e)}")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.0, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    if not os.path.exists("faiss_index/index.faiss"):
        st.error("FAISS index file not found. Please upload and process your PDF files first.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Perform the similarity search
        docs = new_db.similarity_search(user_question,k=3)
        
        # Extract the source metadata for each matched chunk
        matched_sources = []
        for doc in docs:
            matched_sources.append({
                "context": doc.page_content,  # The text chunk
                "doc_name": doc.metadata["doc_name"],  # Document name
                "page_num": doc.metadata["page_num"]  # Page number
            })
        
        # Run the conversational chain with the matched documents
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        # Display the response and the matched context with source info
        st.write("**Reply:** ", response["output_text"])
        st.divider()
        st.divider()
        st.markdown("<h3 style='color: green;'>🔍 Matched Context & Source</h3>", unsafe_allow_html=True)
        for source in matched_sources:
            st.write(f"**Document:** {source['doc_name']} (Page {source['page_num']})")
            st.write(f"**Context:** {source['context']}")
            st.write("---")

    except Exception as e:
        st.error(f"Error loading FAISS index: {str(e)}")

def main():
    pdf_docs = st.sidebar.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", type="pdf", accept_multiple_files=True, key="pdf_uploader")
    
    if st.sidebar.button("Submit & Process", key="process_button") and api_key:  # Check if API key is provided before processing
        with st.spinner("Processing..."):
            # Get PDF text with metadata (document name and page number)
            pdf_metadata = get_pdf_text_with_metadata(pdf_docs)
            
            # Get text chunks with metadata (including document name and page number)
            text_chunks_with_metadata = get_text_chunks_with_metadata(pdf_metadata)
            
            # Build the vector store
            get_vector_store(text_chunks_with_metadata, api_key)
            
            st.sidebar.success("Processing complete! Now you can ask questions.")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question and api_key:  # Ensure API key and user question are provided
        user_input(user_question, api_key)


if __name__ == "__main__":
    main()
