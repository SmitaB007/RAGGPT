# RAGGPT
RAGGPT is an advanced chatbot interface that allows users to upload PDF documents (like research papers) and interact conversationally with their content. Powered by LangChain, ChatGroq’s LLM, and HuggingFace Embeddings, this app provides context-aware responses and maintains memory across the chat session using Retrieval-Augmented Generation (RAG) and session-based history.

# Tech Stack:
Frontend: Streamlit

LLM: Gemma2-9b-It via ChatGroq

Embeddings: all-MiniLM-L6-v2 via HuggingFace

Vector Store: Chroma

PDF Parsing: PyPDFLoader

Memory: ChatMessageHistory

Text Splitting: RecursiveCharacterTextSplitter

# How It Works:
User uploads PDF(s)

PDF content is split into chunks and embedded using HuggingFace embeddings

Chunks are stored in Chroma vector DB

A retriever fetches relevant chunks based on user questions

ChatGroq's Gemma model answers queries using the retrieved context

Conversation history is maintained per session using ChatMessageHistory

# Use Cases:
Research paper analysis

Academic assistant for students

Legal or policy document summarization

PDF knowledge base assistant
