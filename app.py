import streamlit as st
from huggingface_hub import hf_hub_download
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA

st.title("🤖 My Free RAG Chatbot")

# ১. Hugging Face হাব থেকে অটো-ডাউনলোড
@st.cache_resource
def load_llm():
    model_path = hf_hub_download(
        repo_id="unsloth/Llama-3.2-1B-Instruct-GGUF", 
        filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    )
    # ফ্রি ক্লাউড CPU এর জন্য অপ্টিমাইজড কনফিগারেশন
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=1024,      # কনটেক্সট উইন্ডো কিছুটা কমিয়ে RAM বাঁচানো হয়েছে
        n_threads=1,     # ফ্রি কোর এর জন্য ১ রাখা নিরাপদ
        n_batch=128,
        verbose=False
    )
    return llm

llm = load_llm()

# ২. ভেক্টর ডিবি এবং রিট্রিভার সেটআপ
@st.cache_resource
def setup_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    loader = TextLoader("knowledge_base.txt")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    texts = text_splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    return vectorstore.as_retriever(search_kwargs={"k": 2})

retriever = setup_vector_db()
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# ৩. চ্যাট ইন্টারফেস
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
