import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Initialize GROQ chat with provided API key, model name, and settings
groq_api_key = os.environ['GROQ_API_KEY']
llm_groq = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-70b-versatile",
    temperature=0.2
)

# Streamlit UI
st.title("Chat with Your PDF Files")

# File uploader for PDF files
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    texts = []
    metadatas = []
    for file in uploaded_files:
        st.write(f"Processing file: {file.name}")

        # Read the PDF file
        pdf = PyPDF2.PdfReader(file)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
        file_texts = text_splitter.split_text(pdf_text)
        texts.extend(file_texts)

        # Create metadata for each chunk
        file_metadatas = [{"source": f"{i}-{file.name}"} for i in range(len(file_texts))]
        metadatas.extend(file_metadatas)

    # Initialize HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create a Chroma vector store with texts and metadata
    if texts:
        docsearch = Chroma.from_texts(
            texts=texts, 
            embedding=embedding_model,
            metadatas=metadatas
        )

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
            retriever=docsearch.as_retriever(),
            memory=memory,
            return_source_documents=True,
        )

        st.session_state["chain"] = chain
        st.success(f"Processing {len(uploaded_files)} files done. You can now ask questions!")

# Text input for user questions
user_input = st.text_input("Ask a question:")

if user_input:
    if "chain" in st.session_state:
        chain = st.session_state["chain"]

        # Call the chain with user's message content
        res = chain(user_input)
        answer = res["answer"]
        source_documents = res["source_documents"]

        text_elements = []

        # Process source documents if available
        if source_documents:
            for source_idx, source_doc in enumerate(source_documents):
                source_name = f"source_{source_idx}"
                text_elements.append(
                    {"content": source_doc.page_content, "name": source_name}
                )
            source_names = [text_el["name"] for text_el in text_elements]

            # Add source references to the answer
            if source_names:
                answer += f"\nSources: {', '.join(source_names)}"
            else:
                answer += "\nNo sources found"

        # Display results
        st.write(answer)
        for text_el in text_elements:
            st.write(f"Source: {text_el['name']}")
            st.write(text_el['content'])
    else:
        st.warning("No conversation chain is available. Please upload PDF files first.")
