from typing import Dict, Any
from .llm import GeminiLLMWrapper
from backend.prompts import conclusion_prompts
from langchain_core.tools import tool

llm_wrapper = GeminiLLMWrapper()

# Standalone tool function (outside the class)
@tool
def summary(correct: int, total: int, conversation_history: str = "") -> str:
    """Generate a session summary based on student performance and conversation history."""
    prompt = conclusion_prompts.CONCLUSION_TEMPLATE.format(
        correct=correct, 
        total=total, 
        conversation_history=conversation_history
    )
    resp = llm_wrapper.generate_response([{"role":"user","content": prompt}])
    return resp.strip()

class ConclusionAgent:
    def __init__(self, llm=None):
        self.llm = llm or llm_wrapper
        # Reference the standalone tool
        self.summary = summary
    
    def get_all_tools(self):
        """Get all tools as a list for LangGraph integration."""
        return [summary]