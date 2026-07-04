"""
RAG Pipeline - Ollama LLM + nomic-embed-text for embeddings
"""
import io
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

vector_store = None
retriever = None
qa_chain = None
chat_history = []

OLLAMA_URL = "http://host.docker.internal:11434"

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided document context.
Use ONLY the information from the context below to answer the question.
If the answer is not in the context, say "I couldn't find that information in the document."

Context:
{context}

Question: {question}

Answer:"""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def process_pdf(file_bytes: bytes, api_key: str = None) -> dict:
    global vector_store, retriever, qa_chain, chat_history
    chat_history = []

    text = extract_text_from_pdf(file_bytes)
    if not text.strip():
        return {"success": False, "error": "Could not extract text from PDF."}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(text)

    # nomic-embed-text is a dedicated embedding model for Ollama
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_URL
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = OllamaLLM(
        model="llama3.2",
        base_url=OLLAMA_URL
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return {"success": True, "chunks": len(chunks), "characters": len(text)}


def ask_question(question: str) -> dict:
    global qa_chain, chat_history
    if qa_chain is None:
        return {"success": False, "error": "No document loaded. Please upload a PDF first."}

    answer = qa_chain.invoke(question)
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})
    return {"success": True, "answer": answer}


def get_chat_history() -> list:
    return chat_history


def clear_session():
    global vector_store, retriever, qa_chain, chat_history
    vector_store = None
    retriever = None
    qa_chain = None
    chat_history = []
