import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException

#logger init
logger = get_logger(__name__)

#logic to create embeddings using HuggingFace Cloud API
def get_embedding_model():
    try:
        logger.info("Initializing HuggingFace Cloud Embeddings API...")
        
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN is missing. Please add it to your Render dashboard.")

        model = HuggingFaceInferenceAPIEmbeddings(
            api_key=hf_token,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        logger.info("Successfully initialized Cloud Embedding model!")
        return model
    
    except Exception as e:
        error_message = CustomException(f"Error initializing Cloud Embeddings model: {str(e)}")
        logger.error(str(error_message))
        raise error_message