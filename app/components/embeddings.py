from langchain_huggingface import HuggingFaceEmbeddings
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException

#logger init
logger = get_logger(__name__)
#logic to create embeddings using HuggingFaceEmbeddings
def get_embedding_model():
    try:
        logger.info("Initializing HuggingFaceEmbeddings model...")
        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Successfully initialized Embedding model!")
        return model
    
    except Exception as e:
        error_message = CustomException(f"Error initializing HuggingFaceEmbeddings model: {str(e)}")
        logger.error(str(error_message))
        raise error_message
    

    