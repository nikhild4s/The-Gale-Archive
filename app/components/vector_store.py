import os
from langchain_community.vectorstores import FAISS
from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
from app.config.config import DB_FAISS_PATH

#logger init
logger = get_logger(__name__)

def load_vector_store():
    try:
        logger.info("Loading FAISS vector store...")
        embedding_model = get_embedding_model()
        if os.path.exists(DB_FAISS_PATH):
            logger.info(f"FAISS vector store found at {DB_FAISS_PATH}. Loading...")
            return FAISS.load_local(
                    DB_FAISS_PATH, 
                    embedding_model,
                    allow_dangerous_deserialization=True,
                )
        else:
            logger.warning(f"No vector store found at {DB_FAISS_PATH}. Returning None.")
    except Exception as e:
        error_message = CustomException(f"Error loading FAISS vector store: {str(e)}")
        logger.error(str(error_message))
        raise error_message
    

def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No text chunks were found.")
        
        # FIXED: Pulled this entire block back one indentation level
        logger.info("Generating your new vectorstore.")
        embedding_model = get_embedding_model()
        db = FAISS.from_documents(text_chunks, embedding_model)
        logger.info("Saving vectorstore...")
        db.save_local(DB_FAISS_PATH)
        return db
        
    except Exception as e:
        # FIXED: Added the specific error message to the exception
        error_message = CustomException(f"Failed to create new vectorstore: {str(e)}")  
        logger.error(str(error_message))
