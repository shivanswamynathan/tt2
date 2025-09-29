from langchain_core.tools import tool
from .revision_agent import RevisionAgent
from .quiz_agent import QuizAgent  
from .feedback_agent import FeedbackAgent
from .qa_agent import QAAgent
from .conclusion_agent import ConclusionAgent
from .mongodb_client import MongoDBClient
from typing import List, Dict, Any, Optional

# Initialize agents
revision_agent = RevisionAgent()
quiz_agent = QuizAgent()
feedback_agent = FeedbackAgent()
qa_agent = QAAgent()
conclusion_agent = ConclusionAgent()

# MongoDB tools
@tool
def get_topic_subtopics(topic_title: str, mongodb_client: MongoDBClient) -> List[Dict[str, Any]]:
    """Get all subtopics for a specific topic."""
    return mongodb_client.get_topic_subtopics(topic_title)

# Export all tools
def get_all_tools(mongodb_client: MongoDBClient):
    """Get all available tools for the orchestrator."""
    return [
        # Revision tools
        revision_agent.generate_structured_explanation,
        revision_agent.generate_examples,
        revision_agent.make_check_question,
        revision_agent.evaluate_answer,
        revision_agent.handle_qa_request,
        
        # Other agent tools
        conclusion_agent.summary,
        
        # Database tools
        get_topic_subtopics.bind(mongodb_client=mongodb_client),
    ]