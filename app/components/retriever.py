from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from app.components.vector_store import load_vector_store
from app.components.llm import load_llm
from app.common.logger import get_logger
from app.common.custom_exceptions import CustomException
import sys
import traceback

logger = get_logger(__name__)

# THE RE-ENGINEERED PROMPT: Built with Conversational Bypass and Strict Guardrails
CUSTOM_PROMPT_TEMPLATE = """
You are the Clinical AI Analyzer, an intelligent, empathetic, and professional medical AI assistant trained to analyze the Gale Encyclopedia of Medicine.

Context:
{context}

User Query:
{question}

CRITICAL INSTRUCTIONS (EXECUTE IN ORDER):

1. THE CONVERSATIONAL BYPASS: Evaluate the User Query. If it is a casual greeting, expression of gratitude, or general conversation (e.g., "hi", "thanks", "hello", "great job"), COMPLETELY IGNORE the context. Respond naturally, warmly, and briefly as a human-like AI assistant. Do NOT include formatting or a medical disclaimer for casual conversation. Stop here.

2. THE PRIMARY PROTOCOL (RAG): If the User Query is about a medical condition, symptom, or health topic, first prioritize extracting the answer from the provided Context.

3. THE FALLBACK PROTOCOL: If the provided Context DOES NOT contain enough information to answer the query, DO NOT say you cannot find it. Instead, seamlessly fall back on your own general medical knowledge to provide a brief, highly-accurate overview of the topic. Keep this fallback answer concise and professional.

4. FORMATTING & TONE: For medical responses (whether from Context or general knowledge), organize your output logically using clear, bold headings and bullet points to ensure it is highly readable. Maintain an objective, authoritative tone.

5. THE DISCLAIMER: If you provided any medical information, include a brief medical disclaimer at the very end of your response noting this is an AI tool and not a substitute for professional medical advice.

Helpful Answer:
"""

def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

def create_qa_chain():
    try:
        logger.info("Loading vectorstore for your context")
        db = load_vector_store()
        if db is None:
            raise CustomException("Vectorstore is not present or empty")
        
        llm = load_llm()
        if llm is None:
            raise CustomException("LLM not loaded yet")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm = llm,
            chain_type = "stuff",
            # Widen the search to grab the top 6 most relevant chunks of text
            retriever = db.as_retriever(search_kwargs={'k': 6}),
            return_source_documents = False,
            chain_type_kwargs = {'prompt':set_custom_prompt()},
            )
        
        logger.info("Successfully created your QA chain")
        return qa_chain
        
    except Exception as e:
        # Pass the exact error directly back to app.py so Render logs it!
        raise e