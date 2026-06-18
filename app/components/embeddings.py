from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException

#logger init
logger = get_logger(__name__)

#logic to create embeddings using FastEmbed (Zero-Network, Low-RAM)
def get_embedding_model():
    try:
        logger.info("Initializing FastEmbed Local Embeddings...")
        
        # Runs locally without PyTorch, completely bypassing Render's broken DNS
        model = FastEmbedEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        logger.info("Successfully initialized FastEmbed Embedding model!")
        return model
    
    except Exception as e:
        error_message = CustomException(f"Error initializing FastEmbed model: {str(e)}")
        logger.error(str(error_message))
        raise error_message