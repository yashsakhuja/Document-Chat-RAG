import os, tempfile
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import DirectoryLoader
#from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
#from langchain.memory import ConversationBufferMemory
#from langchain.memory.chat_message_histories import StreamlitChatMessageHistory


import streamlit as st

TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')


st.set_page_config(page_title="Document Chat with AI", layout="wide")

st.title("Document Chat with AI")

def input_fields():
    #
    with st.sidebar:
        #
        #if "GOOGLE_API_KEY" in st.secrets:
            #st.session_state.google_api_key = st.secrets.google_api_key
        #else:
        st.session_state.google_api_key = st.text_input("Google API key", type="password")
    #
    st.session_state.source_docs = st.file_uploader(label="Upload PDF File", type="pdf", accept_multiple_files=True)



def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts):
    vectordb = Chroma.from_documents(texts, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.session_state.google_api_key),
                                     persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever

def process_documents():
    if not st.session_state.google_api_key or not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        try:
            for source_doc in st.session_state.source_docs:
                #
                with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                    tmp_file.write(source_doc.read())
                #
                documents = load_documents()
                #
                for _file in TMP_DIR.iterdir():
                    temp_file = TMP_DIR.joinpath(_file)
                    temp_file.unlink()
                #
                texts = split_documents(documents)
                #

                st.session_state.retriever = embeddings_on_local_vectordb(texts)

        except Exception as e:
            st.error(f"An error occurred: {e}")


def query_llm(retriever, query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=st.session_state.google_api_key,temperature=0.0),
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    result = result['answer']
    st.session_state.messages.append((query, result))
    return result


def boot():
    #
    input_fields()
    #
    st.button("Submit Documents", on_click=process_documents)
    #
    if "messages" not in st.session_state:
        st.session_state.messages = []
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    #
    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = query_llm(st.session_state.retriever, query)
        st.chat_message("ai").write(response)

if __name__ == '__main__':
    #
    boot()