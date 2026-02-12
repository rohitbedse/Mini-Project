import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import glob
from dotenv import load_dotenv

# Modern 2026 LangChain Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


load_dotenv()

# Configuration
INDEX_DIR = "faiss_indexes"
PDF_UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(PDF_UPLOAD_DIR, exist_ok=True)

def get_pdf_text(pdf_docs, topic):
    text = ""
    topic_dir = os.path.join(PDF_UPLOAD_DIR, topic)
    os.makedirs(topic_dir, exist_ok=True)
    for pdf in pdf_docs:
        pdf_path = os.path.join(topic_dir, pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(pdf.getbuffer())
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(topic, text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")  # working embedding model
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(os.path.join(INDEX_DIR, f"{topic}_faiss_index"))

def load_vector_store(topic):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")  # same as above
    topic_index_path = os.path.join(INDEX_DIR, f"{topic}_faiss_index")
    try:
        return FAISS.load_local(topic_index_path, embeddings, allow_dangerous_deserialization=True)
    except FileNotFoundError:
        return None

# --- MODERN 2026 CHAIN LOGIC ---
def get_rag_chain(retriever):
    # 1. Define the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    # 2. Create the System Prompt using ChatPromptTemplate
    system_prompt = (
        "Answer the user's question based only on the provided context. "
        "If the answer is not in the context, say 'Answer is not available in the context.' "
        "\n\n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 3. Create the Document Chain (replaces load_qa_chain)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # 4. Create the Retrieval Chain (Modern standard)
    return create_retrieval_chain(retriever, question_answer_chain)

def handle_user_input(topic, user_question):
    vector_store = load_vector_store(topic)
    if not vector_store:
        return "Topic data not found."
    
    # Initialize modern chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    rag_chain = get_rag_chain(retriever)
    
    # Execute using .invoke (2026 LCEL Standard)
    response = rag_chain.invoke({"input": user_question})
    return response["answer"]

# --- STREAMLIT UI ---
def main():
    st.set_page_config(page_title="2026 PDF AI", layout="wide")
    st.header("Chat with PDF (LCEL Architecture) 💁")

    with st.sidebar:
        st.title("Management")
        topic = st.text_input("Topic Name:")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs and topic:
                with st.spinner("Processing..."):
                    text = get_pdf_text(pdf_docs, topic)
                    chunks = get_text_chunks(text)
                    get_vector_store(topic, chunks)
                    st.success("Indexed!")

    index_files = glob.glob(os.path.join(INDEX_DIR, "*_faiss_index"))
    topics = [os.path.basename(f).replace("_faiss_index", "") for f in index_files]
    
    selected_topic = st.selectbox("Select Topic:", options=topics)
    user_question = st.text_input("Ask a question about the topic:")

    if st.button("Ask") and selected_topic and user_question:
        with st.spinner("Thinking..."):
            answer = handle_user_input(selected_topic, user_question)
            st.markdown(f"**AI:** {answer}")

if __name__ == "__main__":
    main()
