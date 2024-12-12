import os
import requests
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS  # Vectorstore DB
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Vector embedding techniques
from dotenv import load_dotenv

load_dotenv()

# Load API keys from .env
groq_api_key = os.getenv("groq_api_key")
os.environ['google_api_key'] = os.getenv("google_api_key")

st.title("Document Research Tool")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-it")
prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Question: {input}""")


def process_document(url):
    """Process a new document URL and update FAISS index."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        # Check if the URL points to a PDF
        if url.lower().endswith(".pdf"):
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open("temp.pdf", "wb") as f:
                    f.write(response.content)
                loader = PyPDFLoader("temp.pdf")
            else:
                raise ValueError("Failed to download the PDF. Please check the URL.")
        else:
            loader = UnstructuredURLLoader(urls=[url])

        # Load and process the document
        docs = loader.load()
        if not docs or len(docs[0].page_content.strip()) == 0:
            raise ValueError("No content could be extracted from the provided URL.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)

        if not final_documents:
            raise ValueError("Document splitting resulted in no content.")

        # Check if FAISS index exists, else create a new one
        if "vectors" in st.session_state:
            st.session_state.vectors.add_documents(final_documents)
        else:
            st.session_state.vectors = FAISS.from_documents(final_documents, embeddings)

        st.session_state.vectors.save_local("faiss_index")
        st.success("FAISS index updated successfully.")

    except Exception as e:
        raise ValueError(f"Error processing the document: {e}")


def load_faiss_index():
    """Load FAISS index if it exists."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        if os.path.exists("faiss_index"):
            st.session_state.vectors = FAISS.load_local("faiss_index", embeddings)
            st.success("FAISS index loaded successfully.")
            return True
        else:
            st.warning("No FAISS index found. Please process a document first.")
            return False
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return False


# Inputs for URL and query
document_url = st.text_input("Enter the document URL (PDF or webpage)")
prompt1 = st.text_input("Enter your query")

# Process the document and save the FAISS index
if st.button("Process Document"):
    if document_url:
        try:
            process_document(document_url)
        except Exception as e:
            st.error(f"Error processing the document: {e}")
    else:
        st.warning("Please provide a document URL.")


# Search functionality
if st.button("Search") and prompt1:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        try:
            import time
            start = time.process_time()
            response = retrieval_chain.invoke({"input": prompt1})

            if "answer" in response and response["answer"]:
                st.write(response["answer"])
            else:
                st.write("No relevant information found for your query.")

            with st.expander("Document Similarity Search"):
                if "context" in response and response["context"]:
                    for i, doc in enumerate(response["context"]):
                        st.write(doc.page_content)
                        st.write("--------------------------------")
                else:
                    st.write("No similar documents found.")
        except Exception as e:
            st.error(f"Error during search: {e}")
    else:
        st.warning("Please process a document or load the FAISS index first.")
