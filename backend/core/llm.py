from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import BaseMessage
from typing import List, Optional
import logging
from backend.config import Config 

logger = logging.getLogger(__name__)

class GeminiLLMWrapper:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=Config.GEMINI_API_KEY,
            model=Config.GEMINI_MODEL,
            temperature=0.3,
            max_output_tokens=4096,  # INCREASED from 2048 to prevent truncation
        )
        logger.info(f"Initialized LLM: {Config.GEMINI_MODEL} with max_tokens=4096")
    
    def generate_response(self, messages: List[BaseMessage], **kwargs) -> str:
        """Generate response synchronously"""
        try:
            logger.debug(f"Generating response for {len(messages)} messages")
            response = self.llm.invoke(messages, **kwargs)
            
            response_length = len(response.content)
            logger.debug(f"Generated response: {response_length} characters")
            
            # Warn if response is suspiciously close to max tokens
            if response_length > 3500:
                logger.warning(f"Response length ({response_length}) is close to max_tokens limit. May be truncated.")
            
            return response.content
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=True)
            return "I apologize, but I'm having trouble generating a response right now."
    
    def generate_response_sync(self, messages: List[BaseMessage], **kwargs) -> str:
        """Alias for generate_response for backward compatibility"""
        return self.generate_response(messages, **kwargs)