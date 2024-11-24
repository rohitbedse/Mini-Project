import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import glob  # For listing FAISS indexes

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Directories for storing data
INDEX_DIR = "faiss_indexes"
PDF_UPLOAD_DIR = "uploaded_pdfs"

# Ensure directories exist
if not os.path.exists(INDEX_DIR):
    os.makedirs(INDEX_DIR)
if not os.path.exists(PDF_UPLOAD_DIR):
    os.makedirs(PDF_UPLOAD_DIR)


# Extract text from uploaded PDFs
def get_pdf_text(pdf_docs, topic):
    text = ""
    topic_dir = os.path.join(PDF_UPLOAD_DIR, topic)
    if not os.path.exists(topic_dir):
        os.makedirs(topic_dir)

    for pdf in pdf_docs:
        # Save the PDF locally for persistence
        pdf_path = os.path.join(topic_dir, pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(pdf.getbuffer())

        # Extract text
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


# Save FAISS vector store from text chunks
def get_vector_store(topic, text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    topic_index_path = os.path.join(INDEX_DIR, f"{topic}_faiss_index")
    vector_store.save_local(topic_index_path)
    return vector_store


# Load FAISS vector store
def load_vector_store(topic):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    topic_index_path = os.path.join(INDEX_DIR, f"{topic}_faiss_index")
    try:
        return FAISS.load_local(topic_index_path, embeddings, allow_dangerous_deserialization=True)
    except FileNotFoundError:
        return None


# List all available topics
def list_available_topics():
    index_files = glob.glob(os.path.join(INDEX_DIR, "*_faiss_index"))
    topics = [os.path.basename(file).replace("_faiss_index", "") for file in index_files]
    return topics


# Create a conversational QA chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "Answer is not available in the context." Don't provide an incorrect answer.

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# Handle user input and generate response
def handle_user_input(topic, user_question):
    vector_store = load_vector_store(topic)
    if not vector_store:
        return "The context data is not available. Please upload and process your PDFs for this topic."
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]


# Streamlit app main function
def main():
    st.set_page_config(page_title="Chat with PDF", layout="wide")
    st.header("Chat with PDF using Gemini 💁")
    st.write("@rohitbedse_")

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.title("Menu:")
        topic = st.text_input("Enter topic name:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs and topic.strip():
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs, topic.strip())
                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(topic.strip(), text_chunks)
                        st.success(f"Processing complete for topic: {topic.strip()}!")
                    else:
                        st.error("Uploaded PDFs are empty or unreadable.")
            else:
                st.warning("Please enter a topic and upload at least one PDF file.")

    # Sidebar for displaying available topics
    st.sidebar.title("Available Topics:")
    topics = list_available_topics()
    if topics:
        st.sidebar.markdown("### Topics:")
        for topic in sorted(topics):  # Display topics in alphabetical order
            st.sidebar.markdown(f"- *{topic}*")
    else:
        st.sidebar.write("No topics available. Please upload PDFs first.")

    # User question input and response display
    st.write("### Ask a Question:")
    selected_topic = st.selectbox("Select a topic:", options=topics)
    user_question = st.text_input("Enter your question:")
    submit_button = st.button("Submit")

    if submit_button:
        if selected_topic and user_question.strip():
            with st.spinner("Generating response..."):
                response = handle_user_input(selected_topic, user_question.strip())
                st.write("### Reply:", response)
        elif not selected_topic:
            st.warning("Please select a topic.")
        elif not user_question.strip():
            st.warning("Please enter a question.")


if __name__ == "__main__":
    main()
