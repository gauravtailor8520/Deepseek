import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import MarkdownHeaderTextSplitter
from docling.document_converter import DocumentConverter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import faiss


# Ensure the vector_db folder exists
VECTOR_DB_FOLDER = "vector_db"
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)


# Load and convert PDF to markdown content
def load_and_convert_document(file_path):
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()


# Split markdown into chunks
def get_markdown_splits(markdown_content):
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    return splitter.split_text(markdown_content)


# Create or load the vector store
def create_or_load_vector_store(filename, chunks, embeddings):
    vector_db_path = Path(VECTOR_DB_FOLDER) / f"{filename}.faiss"

    if vector_db_path.exists():
        vector_store = FAISS.load_local(str(vector_db_path), embeddings=embeddings, allow_dangerous_deserialization=True)
    else:
        single_vector = embeddings.embed_query("initialize")
        index = faiss.IndexFlatL2(len(single_vector))
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        vector_store.add_documents(chunks)
        vector_store.save_local(str(vector_db_path))
    return vector_store


# Build RAG chain
def build_rag_chain(retriever):
    prompt = """
        You are an assistant for financial data analysis. Use the retrieved context to answer questions. 
        If you don't know the answer, say so. 
        Question: {question}
        Context: {context}
        Answer:
    """
    prompt_template = ChatPromptTemplate.from_template(prompt)
    model = ChatOllama(model="deepseek-r1:1.5b", base_url="http://localhost:11434")
    return (
        {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), 
         "question": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )
