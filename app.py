import os
import streamlit as st
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Zyro Dynamics - HR Help Desk", page_icon="ðŸ¢", layout="wide")
st.title("ðŸ¢ Zyro Dynamics Internal HR Portal")
st.markdown("Welcome to the employee self-service help desk. Ask any question regarding company HR policies, benefits, or guidelines.")

@st.cache_resource
def initialize_rag_backend():
    pdf_files = glob.glob("*.pdf") + glob.glob("data/**/*.pdf", recursive=True)
    if not pdf_files:
        return None, None
    
    documents = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        except Exception:
            pass
            
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})
    llm = ChatGroq(temperature=0.1, model_name="llama-3.1-8b-instant", groq_api_key=st.secrets["GROQ_API_KEY"])
    
    return retriever, llm

retriever, llm = initialize_rag_backend()

if llm and retriever:
    OOS_PROMPT = ChatPromptTemplate.from_messages([
        ("system", "You are a strict classifier. Determine if the user's question is related to company HR policies, employee benefits, IT security, workplace rules, or onboarding. Answer ONLY with the word 'YES' if it is related, or 'NO' if it is entirely unrelated."),
        ("human", "{question}")
    ])
    classifier_chain = OOS_PROMPT | llm | StrOutputParser()

    template = (
        "You are an HR Help Desk Assistant for Zyro Dynamics.\n"
        "Answer employee questions using ONLY the provided context from internal company HR policy documents.\n\n"
        "Context: {context}\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    RAG_PROMPT = ChatPromptTemplate.from_template(template)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask a question about Zyro Dynamics policies..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if not (llm and retriever):
            response = "System initialization error. Please verify database configurations."
            st.markdown(response)
        else:
            with st.spinner("Analyzing company policies..."):
                classification = classifier_chain.invoke({"question": user_query}).strip().upper()
                
                if "NO" in classification:
                    response = "I can only answer HR-related questions from Zyro Dynamics policy documents."
                    st.markdown(response)
                else:
                    matched_docs = retriever.invoke(user_query)
                    context_str = "\n\n".join(doc.page_content for doc in matched_docs)
                    qa_prompt = RAG_PROMPT.format(context=context_str, question=user_query)
                    response = llm.predict(qa_prompt)
                    st.markdown(response)
                    
                    with st.expander("ðŸ“š View Sourced Policy Context"):
                        for i, doc in enumerate(matched_docs, 1):
                            source_name = os.path.basename(doc.metadata.get("source", "Policy Document"))
                            page_num = doc.metadata.get("page", 0) + 1
                            st.markdown(f"**Source {i}:** {source_name} (Page {page_num})")
                            st.caption(doc.page_content)
                            st.markdown("---")

    st.session_state.messages.append({"role": "assistant", "content": response})
