# streamlit run app.py

import os
import streamlit as st
from pathlib import Path
from rag import load_and_convert_document, get_markdown_splits, create_or_load_vector_store, build_rag_chain
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from pdf2image import convert_from_path, exceptions
from PIL import Image

# Path to vector DB folder
VECTOR_DB_FOLDER = "vector_db"
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

# Set environment variable to avoid KMP duplicate library error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Function to display PDF content as images in the sidebar
def display_pdf_in_sidebar(pdf_path, file_name):
    try:
        images_folder = Path(VECTOR_DB_FOLDER) / file_name / "images"
        os.makedirs(images_folder, exist_ok=True)

        # Check if images already exist
        image_paths = list(images_folder.glob("*.png"))
        if image_paths:
            # If images exist, display them
            for img_path in image_paths:
                image = Image.open(img_path)
                st.sidebar.image(image, caption=f"Page {image_paths.index(img_path) + 1}", use_container_width=True)
        else:
            # Convert PDF to images (one per page)
            images = convert_from_path(pdf_path)  # This will render all pages by default
            for i, image in enumerate(images):
                img_path = images_folder / f"page_{i + 1}.png"
                image.save(img_path, "PNG")  # Save image to disk
                st.sidebar.image(image, caption=f"Page {i + 1}", use_container_width=True)

    except exceptions.PDFPageCountError:
        st.sidebar.error("Error: Unable to get page count. The PDF may be corrupted or empty.")
    except exceptions.PDFSyntaxError:
        st.sidebar.error("Error: PDF syntax is invalid or the document is corrupted.")
    except Exception as e:
        st.sidebar.error(f"Error loading PDF: {str(e)}")

# Streamlit title and layout
st.title("Financial Documents RAG With DeepSeek-R1 🐟 ")

# Dropdown to select vector DB or upload a new document
vector_db_options = [f.stem for f in Path(VECTOR_DB_FOLDER).glob("*.faiss")]
vector_db_options.append("Upload New Document")  # Add option to upload a new document
selected_vector_db = st.selectbox("Select Vector DB or Upload New Document", vector_db_options, index=0)

# If 'Upload New Document' is selected, show the file uploader
if selected_vector_db == "Upload New Document":
    uploaded_file = st.file_uploader("Upload a PDF file for analysis", type=["pdf"])

    # Process the uploaded PDF
    if uploaded_file:
        st.sidebar.subheader("Uploaded PDF")
        st.sidebar.write(uploaded_file.name)

        # Save the PDF file temporarily and display it
        temp_path = f"temp_{uploaded_file.name}"
        document_binary = uploaded_file.read()
        with open(temp_path, "wb") as f:
            f.write(document_binary)

        # Display PDF in the sidebar (show all pages)
        display_pdf_in_sidebar(temp_path, uploaded_file.name.split('.')[0])

        # PDF processing button
        if st.button("Process PDF and Store in Vector DB"):
            with st.spinner("Processing document..."):
                # Convert PDF to markdown directly
                markdown_content = load_and_convert_document(temp_path)
                chunks = get_markdown_splits(markdown_content)

                # Initialize embeddings
                embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")

                # Create or load vector DB and store PDF along with it
                vector_store = create_or_load_vector_store(uploaded_file.name.split(".")[0], chunks, embeddings)

                # Ensure vector DB and PDF are stored correctly
                vector_db_path = Path(VECTOR_DB_FOLDER) / f"{uploaded_file.name.split('.')[0]}.faiss"
                vector_store.save_local(str(vector_db_path))  # Save FAISS vector store

                # Store the PDF file alongside the vector DB
                pdf_path = Path(VECTOR_DB_FOLDER) / f"{uploaded_file.name}"
                with open(pdf_path, "wb") as f:
                    f.write(document_binary)

                st.success("PDF processed and stored in the vector database.")

                # Clean up the temporary file
                Path(temp_path).unlink()

elif selected_vector_db != "Upload New Document":
    # Load the selected vector DB
    vector_db_path = Path(VECTOR_DB_FOLDER) / f"{selected_vector_db}.faiss"
    if vector_db_path.exists():
        embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
        vector_store = FAISS.load_local(str(vector_db_path), embeddings=embeddings, allow_dangerous_deserialization=True)

        # Display PDF in the sidebar
        pdf_path = Path(VECTOR_DB_FOLDER) / f"{selected_vector_db}.pdf"
        if pdf_path.exists():
            display_pdf_in_sidebar(pdf_path, selected_vector_db)
        else:
            st.sidebar.warning("PDF file not found for the selected vector DB.")
    else:
        st.sidebar.warning(f"Vector DB '{selected_vector_db}' not found.")

# Question input section
question = st.text_input("Enter your question:", placeholder="e.g., What is the company's revenue for the quarter?")

# Button to process and generate answers
if st.button("Submit Question") and question and selected_vector_db != "Upload New Document":
    with st.spinner("Answering your question..."):
        # Build retriever from the selected vector store
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 5})

        # Build and run the RAG chain
        rag_chain = build_rag_chain(retriever)

        # Create a placeholder for streaming response
        response_placeholder = st.empty()  # Create an empty placeholder for the answer

        # Stream the response as it is generated
        response = ""
        for chunk in rag_chain.stream(question):
            response += chunk  # Append each chunk of the response
            response_placeholder.markdown(response.replace('$', '\\$'))  # Update the placeholder with the new response
