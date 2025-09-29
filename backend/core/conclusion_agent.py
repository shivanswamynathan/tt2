from typing import Dict, Any
from .llm import GeminiLLMWrapper
from backend.prompts import conclusion_prompts
from langchain_core.tools import tool

llm_wrapper = GeminiLLMWrapper()

@tool
def summary(correct: int, total: int, conversation_history: str = "") -> str:
    """Generate a session summary based on student performance and conversation history."""
    if not isinstance(correct, int) or not isinstance(total, int):
        raise ValueError("correct and total must be integers")
    if total < 0 or correct < 0 or correct > total:
        raise ValueError("Invalid values: correct and total must be non-negative, and correct cannot exceed total")
    if not conversation_history:
        conversation_history = "No conversation history available."
    
    prompt = conclusion_prompts.CONCLUSION_TEMPLATE.format(
        correct=correct, 
        total=total, 
        conversation_history=conversation_history
    )
    try:
        resp = llm_wrapper.generate_response([{"role": "user", "content": prompt}])
        if not isinstance(resp, str) or not resp.strip():
            raise ValueError("Invalid response from LLM")
        return resp.strip()
    except Exception as e:
        return f"Session completed. You mastered {correct} out of {total} concepts."

class ConclusionAgent:
    def __init__(self, llm=None):
        self.llm = llm or llm_wrapper
        self.summary = summary
    
    def get_all_tools(self):
        """Get all tools as a list for LangGraph integration."""
        return [self.summary]