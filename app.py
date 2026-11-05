import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.title("🤖 IIUC Admission RAG Chatbot")

# 1. Check API Key
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("⚠️ Please add your GROQ_API_KEY in Streamlit Secrets!")
    st.stop()

# 2. Load LLM
@st.cache_resource
def load_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

llm = load_llm()

# 3. Vector DB & Retriever Setup
@st.cache_resource
def setup_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    pdf_filename = "59Online-Application-Instruction-Spring-2027-ENG.pdf"
    
    if not os.path.exists(pdf_filename):
        st.error(f"❌ '{pdf_filename}' file not found in the repository!")
        st.stop()
        
    loader = PyPDFLoader(pdf_filename)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # In-memory Chroma database to avoid write permission errors on cloud platforms
    vectorstore = Chroma.from_documents(texts, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = setup_vector_db()

# 4. RAG Chain Setup
system_prompt = (
    "You are an expert academic assistant for International Islamic University Chittagong (IIUC). "
    "Use the following pieces of retrieved context to answer the question accurately. "
    "If you don't know the answer, politely say that you don't know.\n\n"
    "Context:\n{context}"
)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 5. UI Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("IIUC অ্যাডমিশন নিয়ে যেকোনো প্রশ্ন করুন..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})