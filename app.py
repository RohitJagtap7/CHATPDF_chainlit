import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv() 

# Function to initialize conversation chain with GROQ language model
groq_api_key = os.environ['GROQ_API_KEY']

# Initialize GROQ chat with provided API key, model name, and settings
llm_groq = ChatGroq(
    groq_api_key=groq_api_key, 
    model_name="llama-3.1-70b-versatile",
    temperature=0.2
)

@cl.on_chat_start
async def on_chat_start():
    files = None  # Initialize variable to store uploaded files

    # Wait for the user to upload files
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload one or more PDF files to begin!",
            accept=["application/pdf", "image/jpeg", "image/png"],  # Accept PDFs and images
            max_size_mb=100,  # Optionally limit the file size,
            max_files=10,
            timeout=180,  # Set a timeout for user response
        ).send()

    # Process each uploaded file
    texts = []
    metadatas = []
    for file in files:
        print(f"Processing file: {file.name}")  # Print the file name for debugging

        # Check if the file is a PDF or an image
        if file.name.endswith(".pdf"):
            # Read the PDF file
            pdf = PyPDF2.PdfReader(file.path)
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

        elif file.name.endswith((".jpg", ".jpeg", ".png")):
            # Process image files (You might need to implement specific handling for images)
            print(f"Image file {file.name} uploaded, but no processing is defined for images yet.")
            # Implement any specific processing if required

    # Initialize HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create a Chroma vector store with texts and metadata, passing the embedding function directly
    if texts:  # Ensure there are texts to create the vector store
        docsearch = Chroma.from_texts(
            texts=texts, 
            embedding=embedding_model,  # Directly pass the embedding function
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

        # Store the chain in user session
        cl.user_session.set("chain", chain)

        # Inform the user that processing has ended. You can now chat.
        msg = cl.Message(content=f"Processing {len(files)} files done. You can now ask questions!")
        await msg.send()
    else:
        msg = cl.Message(content="No PDF files were processed. Please upload PDF files to proceed.")
        await msg.send()

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from user session
    chain = cl.user_session.get("chain")
    if chain:
        # Callbacks happen asynchronously/parallel 
        cb = cl.AsyncLangchainCallbackHandler()

        # Call the chain with user's message content
        res = await chain.ainvoke(message.content, callbacks=[cb])
        answer = res["answer"]
        source_documents = res["source_documents"]

        text_elements = []  # Initialize list to store text elements

        # Process source documents if available
        if source_documents:
            for source_idx, source_doc in enumerate(source_documents):
                source_name = f"source_{source_idx}"
                # Create the text element referenced in the message
                text_elements.append(
                    cl.Text(content=source_doc.page_content, name=source_name)
                )
            source_names = [text_el.name for text_el in text_elements]

            # Add source references to the answer
            if source_names:
                answer += f"\nSources: {', '.join(source_names)}"
            else:
                answer += "\nNo sources found"

        # Return results
        await cl.Message(content=answer, elements=text_elements).send()
    else:
        await cl.Message(content="No conversation chain is available. Please upload PDF files first.").send()
