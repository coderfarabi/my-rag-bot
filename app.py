import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA

st.title("🤖 My Fast Online RAG Chatbot")

# ১. সরাসরি Groq API Key সেট করা (আপনার কি-টি এখানে বসান)
# আরও নিরাপদ করতে Streamlit Secrets ব্যবহার করা যায়, তবে শুরুতে সরাসরি বসালেও সমস্যা নেই
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ২. অনলাইন ক্লাউড LLM লোড করা (এটি চোখের পলকে সুপার ফাস্ট কাজ করবে)
@st.cache_resource
def load_llm():
    # Llama 3.3 বা 3.1 এর এই মডেলটি গ্রকে সম্পূর্ণ ফ্রি এবং এর লিমিট অনেক বেশি
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
