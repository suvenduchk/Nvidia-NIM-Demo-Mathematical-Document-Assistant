import streamlit as st
import os
import tempfile
import fitz  # PyMuPDF
import io
import re
import time
import shutil
import glob
from PIL import Image
import pytesseract
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

## load the NVIDIA API key
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
if nvidia_api_key:
    os.environ['NVIDIA_API_KEY'] = nvidia_api_key
else:
    st.error("NVIDIA_API_KEY environment variable is not set. Please set it in your .env file or environment.")

# Store temp directory path for cleanup
if 'temp_dirs' not in st.session_state:
    st.session_state.temp_dirs = []

# Check if Tesseract is installed
try:
    pytesseract.get_tesseract_version()
    TESSERACT_INSTALLED = True
    st.success("Tesseract OCR is available")
except Exception as e:
    TESSERACT_INSTALLED = False
    st.warning("Tesseract OCR is not installed. Text extraction from scanned PDFs may not work.")
    st.info("Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")

def extract_text_with_ocr(page):
    """Extract text from a PDF page using OCR"""
    try:
        # Convert the page to an image
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # Increase resolution for better OCR
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        
        # Use OCR to extract text
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        st.error(f"OCR error: {str(e)}")
        return ""

def extract_text_from_pdf(file_path):
    """Extract text from PDF using PyMuPDF and OCR if needed"""
    try:
        doc = fitz.open(file_path)
        text_list = []
        total_pages = len(doc)
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_estimate = st.empty()
        
        # For time estimation
        start_time = time.time()
        page_times = []
        
        for page_num, page in enumerate(doc):
            page_start_time = time.time()
            
            # Update progress bar
            progress = (page_num) / total_pages
            progress_bar.progress(progress)
            status_text.text(f"Processing page {page_num+1} of {total_pages}")
            
            # Try normal text extraction first
            text = page.get_text()
            
            # If no text is found and OCR is available, try OCR
            if not text.strip() and TESSERACT_INSTALLED:
                status_text.text(f"Processing page {page_num+1} of {total_pages} (Using OCR)")
                text = extract_text_with_ocr(page)
            
            # Create document with extracted text or placeholder
            text_list.append(
                Document(
                    page_content=text if text.strip() else f"Page {page_num+1} (No extractable text)",
                    metadata={"source": os.path.basename(file_path), "page": page_num + 1}
                )
            )
            
            # Calculate time per page for estimation
            page_time = time.time() - page_start_time
            page_times.append(page_time)
            
            # Estimate remaining time
            if page_num > 0:
                avg_time_per_page = sum(page_times) / len(page_times)
                remaining_pages = total_pages - (page_num + 1)
                est_remaining_time = avg_time_per_page * remaining_pages
                time_estimate.text(f"Estimated time remaining: {est_remaining_time:.1f} seconds")
        
        # Complete the progress bar
        progress_bar.progress(1.0)
        status_text.text(f"Completed processing {total_pages} pages in {time.time() - start_time:.1f} seconds")
        time_estimate.empty()
        
        # Show sample of first page
        if text_list and text_list[0].page_content:
            preview = text_list[0].page_content[:100] if len(text_list[0].page_content) > 100 else text_list[0].page_content
            st.info(f"Sample text from first page: {preview}")
        
        # Close the document to free resources
        doc.close()
        
        return text_list
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return []

def cleanup_temp_dirs():
    """Clean up temporary directories"""
    for temp_dir in st.session_state.temp_dirs:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as e:
            pass
    
    # Reset the list
    st.session_state.temp_dirs = []

def vector_embedding(uploaded_files=None):
    if "vectors" not in st.session_state or uploaded_files:
        # Clean up any previous temp directories
        cleanup_temp_dirs()
        
        with st.spinner("Processing documents..."):
            try:
                st.session_state.embeddings = NVIDIAEmbeddings()
                
                # If user uploaded files, use those instead of the default directory
                if uploaded_files:
                    # Create a temporary directory to store uploaded files
                    temp_dir = tempfile.mkdtemp()
                    # Store for later cleanup
                    st.session_state.temp_dirs.append(temp_dir)
                    
                    docs = []
                    
                    # Save uploaded files to temp directory and load them
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Extract text from PDF
                        st.subheader(f"Processing {uploaded_file.name}")
                        file_docs = extract_text_from_pdf(file_path)
                        
                        # Add documents to collection
                        if file_docs:
                            docs.extend(file_docs)
                            st.success(f"Successfully processed {len(file_docs)} pages from {uploaded_file.name}")
                    
                    st.session_state.docs = docs
                else:
                    st.error("No files uploaded")
                    return False
                
                # Check if we have any documents
                if not st.session_state.docs:
                    st.error("No documents were loaded. Please check your PDF files.")
                    return False
                
                # Debug info
                st.info(f"Loaded {len(st.session_state.docs)} document pages in total")
                
                # Use a smaller chunk size for better processing
                st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, 
                    chunk_overlap=50,
                    separators=["\n\n", "\n", " ", ""]
                )
                
                with st.spinner("Creating text chunks..."):
                    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
                
                # Debug info
                st.info(f"Created {len(st.session_state.final_documents)} text chunks")
                
                # Check if we have any chunks after splitting
                if not st.session_state.final_documents:
                    st.error("No text could be extracted from the documents.")
                    return False
                
                with st.spinner("Creating vector embeddings..."):
                    try:
                        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
                        return True
                    except Exception as e:
                        st.error(f"Error creating vector store: {str(e)}")
                        return False
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                return False
    return False

def render_math_diagrams(text):
    """Enhanced function to render mathematical diagrams and formulas"""
    if not text or not text.strip():
        st.write("📝 **Content:** No content available")
        return
    
    # Look for LaTeX blocks (enclosed in $ or \begin{equation})
    latex_blocks = re.findall(r'\$\$(.*?)\$\$|\\begin\{equation\}(.*?)\\end\{equation\}', text, re.DOTALL)
    
    if latex_blocks:
        # Text contains LaTeX blocks, render them separately
        parts = re.split(r'\$\$(.*?)\$\$|\\begin\{equation\}(.*?)\\end\{equation\}', text, flags=re.DOTALL)
        for i, part in enumerate(parts):
            if i % 3 == 0:  # Regular text
                if part and part.strip():
                    st.markdown(f"**📖 Explanation:** {part.strip()}")
            elif i % 3 == 1 or i % 3 == 2:  # LaTeX block
                if part and part.strip():
                    st.markdown("**🔢 Mathematical Expression:**")
                    st.latex(part)
    else:
        # No explicit LaTeX blocks, show as regular content
        st.markdown(f"**📝 Content:** {text.strip()}")


st.title("📚 Nvidia NIM Demo - Mathematical Document Assistant")

# Add info about math rendering
st.info("""
💡 **Features:**
- Upload multiple PDF files for analysis
- Advanced OCR for scanned documents
- LaTeX rendering for mathematical formulas
- AI-powered question answering
""")

# File uploader for PDFs
st.subheader("📁 Upload Your Documents")
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True, help="Select one or more PDF files to analyze")

# Button to process uploaded files
if st.button("🚀 Process Uploaded Documents", type="primary"):
    if uploaded_files:
        success = vector_embedding(uploaded_files)
        if success:
            st.success(f"✅ Successfully processed {len(uploaded_files)} PDF files")
    else:
        st.warning("⚠️ Please upload PDF files first")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.

For mathematical formulas, equations, or expressions, format them using proper LaTeX notation.
- Use $$....$$ to enclose standalone mathematical expressions
- For inline math, use $...$ 
- For complex equations, use \\begin{{equation}}...\\end{{equation}}
- Use \\frac{{numerator}}{{denominator}} for fractions
- Use \\sqrt{{x}} for square roots
- Use ^{{power}} for exponents and _{{subscript}} for subscripts
- Use \\sum, \\int, \\prod for summations, integrals, and products
- For matrices, use \\begin{{matrix}}...\\end{{matrix}} with \\\\ for row breaks and & for column separators

If the answer involves diagrams, describe them clearly using LaTeX notation where possible.

<context>
{context}
</context>
Question: {input}
"""
)

st.subheader("❓ Ask Questions About Your Documents")
prompt1 = st.text_input("Enter your question:", placeholder="e.g., What is the quadratic formula?", help="Ask any question about the content in your uploaded PDFs")

if prompt1:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please process documents first before asking questions")
    else:
        try:
            document_chain = create_stuff_documents_chain(ChatNVIDIA(model="meta/llama3-70b-instruct"), prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            with st.spinner("🤔 Generating answer..."):
                start = time.time()
                response = retrieval_chain.invoke({'input': prompt1})
                end_time = time.time() - start
                
                # Display the answer with enhanced LaTeX rendering for math expressions
                st.subheader("💡 Answer:")
                render_math_diagrams(response['answer'])
                
                st.info(f"⏱️ Response time: {end_time:.1f} seconds")

                # With a streamlit expander
                with st.expander("🔍 Document Similarity Search - Source References"):
                    st.markdown("**📚 Related content from your documents:**")
                    # Find the relevant chunks
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**📄 Reference {i+1}:**")
                        render_math_diagrams(doc.page_content)
                        st.divider()
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")

# Clean up temp files when the app is done
if st.session_state.get('temp_dirs'):
    cleanup_temp_dirs()