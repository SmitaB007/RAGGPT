import os
from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

import streamlit as st 

from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory

st.title("Conversational RAG with docs and chat history")
st.write("Upload PDF and chat with content")

api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")
session_id=st.text_input("Session ID",value="default")
if 'store' not in st.session_state:
    st.session_state.store={}  #created store if not present

uploaded_files=st.file_uploader("choose a pdf file",type="pdf",accept_multiple_files=True)

if uploaded_files:
    documents=[]
    for f in uploaded_files:
        temp = f"./temp.pdf"
        with open(temp,"wb") as fi:
             fi.write(f.getvalue())
             fi_name=f.name

        loader=PyPDFLoader(temp)
        docs=loader.load()
        documents.extend(docs)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=200)
    splits=text_splitter.split_documents(documents)
    vectorstorev=Chroma.from_documents(documents=splits,embedding=embeddings)
    retriever=vectorstorev.as_retriever()

    contextualize_q_system_prompt=(
        "Given a chat history and the latest user question" \
        "which might reference context in the chat history." \
        "formulate a standalone question which can be understood"
        "without the chat history, Do not answer the question," \
        "just reformulate it if needed and otherwise return it as it is"
        )
    
    contextualize_q_prompt=ChatPromptTemplate.from_messages(
        [
            ("system",contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )
    
    history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

    system_prompt=(
        "You are an assistant for question-answering tasks." \
        "Use the following pieces of retrieved context to answer" \
        "the question.if you don't know the answer,say thank you" \
        "don't know. Use three sentences maximum and keep the" \
        "answer concise." \
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )
    qa_chain =create_stuff_documents_chain(llm,qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever,qa_chain)
    
    def get_session_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]
    
    conversational_rag_chain=RunnableWithMessageHistory(
        rag_chain,get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    user_input = st.text_input("your question:")
    if user_input:
        session_history=get_session_history(session_id)
        res=conversational_rag_chain.invoke(
            {"input":user_input},
            config={
                "configurable":{"session_id":session_id}
            }
        )
        # st.write(st.session_state.store)
        st.write("Assistant:",res['answer'])
        st.write("Chat History:",session_history.messages)
