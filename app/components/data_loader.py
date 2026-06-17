import os
from app.components.load_pdf import load_pdf_files ,create_text_chunks
from app.components.vector_store import load_vector_store, save_vector_store
from app.config.config import DB_FAISS_PATH
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException

logger = get_logger(__name__)

def process_and_store_pdf():
    try:
        logger.info("Making the vectorstore....")

        documents = load_pdf_files()
        text_chunks = create_text_chunks(documents)
        save_vector_store(text_chunks) 
        logger.info("Vectorstore created and saved successfully!")

    except Exception as e:
        error_message = CustomException(f"Failed to create new vectorstore: {str(e)}")
        logger.error(str(error_message))

if __name__== "__main__":
    process_and_store_pdf()