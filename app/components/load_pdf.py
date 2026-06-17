import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.config.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

#logger init
logger = get_logger(__name__)

#logic to load pdf files and split them into chunks
def load_pdf_files():
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException(f"Data path {DATA_PATH} does not exist.")
        logger.info(f"Loading PDF files from {DATA_PATH}...")

        loader = DirectoryLoader(
            DATA_PATH, 
            glob="*.pdf", # FIXED: Changed from "pdf" to "*.pdf" to match file extensions
            loader_cls=PyPDFLoader
        )
        documents = loader.load()

        if not documents:
            logger.warning("No PDF files found in the specified directory.")
        else:
            logger.info(f"Successfully loaded {len(documents)} PDF files!")
        return documents

    except Exception as e:
        error_message = CustomException(f"Error loading PDF files: {str(e)}")
        logger.error(str(error_message))
        return []
    
# FIXED: Added 'documents' as a parameter to match data_loader.py
def create_text_chunks(documents): 
    try:
        if not documents:
            raise CustomException("No documents to process for text chunking.")
        logger.info(f"Splitting {len(documents)} documents into chunks...")

        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, 
                chunk_overlap=CHUNK_OVERLAP
            )
        text_chunks = text_splitter.split_documents(documents)
        logger.info(f"Successfully created {len(text_chunks)} text chunks!")
        return text_chunks
    except Exception as e:
        error_message = CustomException(f"Error creating text chunks: {str(e)}")
        logger.error(str(error_message))
        return []