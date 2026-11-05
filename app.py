import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader  # আপনার জুপিটার কোড থেকে নেওয়া
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.title("🤖 IIUC Admission RAG Chatbot")

# ১. স্ট্রিমলিট সিক্রেটস থেকে API Key চেক ও লোড করা
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("⚠️ Please add your GROQ_API_KEY in Streamlit Secrets!")
    st.stop()

# ২. অনলাইন ক্লাউড LLM লোড করা (Groq এর ফ্রি এবং সুপার ফাস্ট Llama)
@st.cache_resource
def load_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

llm = load_llm()

# ৩. ভেক্টর ডিবি এবং রিট্রিভার সেটআপ (আপনার PDF ফাইলটি রিড করবে)
@st.cache_resource
def setup_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # আপনার জুপিটার নোটবুকের পিডিএফ ফাইলের নাম এখানে দেওয়া হয়েছে
    pdf_filename = "59Online-Application-Instruction-Spring-2027-ENG.pdf"
    
    if not os.path.exists(pdf_filename):
        st.error(f"❌ '{pdf_filename}' file not found in the GitHub repository!")
        st.stop()
        
    loader = PyMuPDFLoader(pdf_filename)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever = setup_vector_db()

# ৪. আধুনিক চ্যাট প্রম্পট এবং RAG চেইন তৈরি
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

# ৫. চ্যাট ইন্টারফেস (Streamlit UI)
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
