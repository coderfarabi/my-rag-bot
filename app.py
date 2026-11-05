import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.title("🤖 IIUC Admission RAG Chatbot")

# ১. API Key চেক ও লোড করা
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("⚠️ Please add your GROQ_API_KEY in Streamlit Secrets!")
    st.stop()

# ২. LLM লোড করা
@st.cache_resource
def load_llm():
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)

llm = load_llm()

# ৩. ভেক্টর ডিবি এবং রিট্রিভার সেটআপ (ফোল্ডার থেকে সমস্ত PDF পড়ার জন্য)
@st.cache_resource
def setup_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # PDF ফাইলগুলো রাখা ফোল্ডারের নাম (যেমন: 'docs' বা 'pdfs')
    pdf_folder_path = "docs"
    
    if not os.path.exists(pdf_folder_path):
        st.error(f"❌ Folder '{pdf_folder_path}' not found in the repository!")
        st.stop()
        
    # PyPDFDirectoryLoader দিয়ে ফোল্ডারের ভেতরের সব PDF একসাথে লোড হবে
    loader = PyPDFDirectoryLoader(pdf_folder_path)
    documents = loader.load()
    
    if not documents:
        st.error(f"⚠️ No PDF files found inside '{pdf_folder_path}' folder!")
        st.stop()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(texts, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})

retriever = setup_vector_db()

# ৪. ডকুমেন্টস প্রসেস করার ফাংশন
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ৫. চ্যাট প্রম্পট এবং আধুনিক LCEL RAG চেইন তৈরি
system_prompt = (
    "You are an expert academic assistant for International Islamic University Chittagong (IIUC). "
    "Use the following pieces of retrieved context to answer the question accurately. "
    "If you don't know the answer, politely say that you don't know.\n\n"
    "Context:\n{context}"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}"),
])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

# ৬. চ্যাট ইন্টারফেস (Streamlit UI)
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
            answer = rag_chain.invoke(prompt)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})