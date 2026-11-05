import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
# ল্যাংচেইনের নতুন ভার্সন অনুযায়ী সংশোধিত ইম্পোর্ট
from langchain.chains.retrieval_qa.base import RetrievalQA

st.title("🤖 My Fast Online RAG Chatbot")

# ১. স্ট্রিমলিট সিক্রেটস থেকে API Key লোড করা (সবচেয়ে নিরাপদ)
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("Please add your GROQ_API_KEY in Streamlit Secrets!")
    st.stop()

# ২. অনলাইন ক্লাউড LLM লোড করা
@st.cache_resource
def load_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

llm = load_llm()

# ৩. ভেক্টর ডিবি এবং রিট্রিভার সেটআপ
@st.cache_resource
def setup_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    loader = TextLoader("knowledge_base.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    return vectorstore.as_retriever(search_kwargs={"k": 2})

retriever = setup_vector_db()
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# ৪. চ্যাট ইন্টারফেস
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({"query": prompt})
            answer = response["result"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
