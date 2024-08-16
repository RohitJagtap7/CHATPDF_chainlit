import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

# Function to initialize conversation chain with GROQ language model
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize GROQ chat with provided API key, model name, and settings
llm_groq = ChatGroq(
    groq_api_key=groq_api_key, 
    model_name="llama-3.1-70b-versatile",
    temperature=0.2
)

# Streamlit UI setup
st.set_page_config(page_title="Document Q&A", page_icon=":books:", layout="wide")
st.title("📄 Document Q&A")

# Sidebar file uploader
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Please upload PDF files to begin",
        type=["pdf"],
        accept_multiple_files=True
    )

# Initialize texts and metadata for processing
texts = []
metadatas = []

# Check if files were uploaded and if they haven't been processed before
if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.spinner(f"Processing file: {uploaded_file.name}..."):
            # Read the PDF file
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()

            # Split the text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
            file_texts = text_splitter.split_text(pdf_text)
            texts.extend(file_texts)

            # Create metadata for each chunk
            file_metadatas = [{"source": f"{i}-{uploaded_file.name}"} for i in range(len(file_texts))]
            metadatas.extend(file_metadatas)

    # Initialize HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a Chroma vector store with texts and metadata, passing the embedding function directly
    if texts:
        docsearch = Chroma.from_texts(
            texts=texts, 
            embedding=embedding_model, 
            metadatas=metadatas
        )

        # Store the vector store in session state
        st.session_state.docsearch = docsearch

        # Initialize message history for conversation
        message_history = ChatMessageHistory()

        # Memory for conversational context
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )

        # Create a chain that uses the Chroma vector store
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm_groq,
            chain_type="stuff",
            retriever=st.session_state.docsearch.as_retriever(),
            memory=memory,
            return_source_documents=True,
        )

        # Store the chain in session state
        st.session_state.chain = chain

        st.success(f"Processing {len(uploaded_files)} files done. You can now ask questions!")

# Input field always active
st.markdown("---")
user_question = st.text_input("Ask a question based on the uploaded documents:", key="user_input", help="Type your question here and press Enter.")

# Display chat history and allow questions even if no new files are uploaded
if "chain" in st.session_state and user_question:
    with st.spinner("Generating response..."):
        res = st.session_state.chain(user_question)
        answer = res["answer"]
        source_documents = res["source_documents"]

        # Display the answer
        st.write(f"**Answer:** {answer}")

        # Display the source documents if available
        if source_documents:
            st.write("### Sources")
            for source_idx, source_doc in enumerate(source_documents):
                with st.expander(f"Source {source_idx+1}"):
                    st.write(source_doc.page_content)
elif not uploaded_files:
    st.info("Please upload PDF files to ask questions based on them.")
