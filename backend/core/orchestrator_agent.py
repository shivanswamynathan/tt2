import asyncio
from typing import TypedDict,List, Dict, Any, Optional
from .revision_agent import RevisionAgent
from backend.core.quiz_agent import QuizAgent
from backend.core.feedback_agent import FeedbackAgent
from backend.core.qa_agent import QAAgent
from backend.core.conclusion_agent import ConclusionAgent
from .mongodb_client import MongoDBClient
from backend.models.schemas import RevisionSessionData, SessionState
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# LangGraph import - we use it to express the orchestrator flow
from langgraph.graph import StateGraph

class OrchestratorState(TypedDict, total=False):
    user_message: Optional[str]
    assistant_message: Optional[str]
    stage: Optional[str]
    conversation_history: Optional[List[Dict[str, Any]]]
    current_chunk_index: Optional[int]
    current_question_concept: Optional[str]
    current_expected_keywords: Optional[List[str]]
    expecting_answer: Optional[bool]

class OrchestratorAgent:
    """
    Main orchestrator with quiz-based progression: 5 correct answers required per concept
    """
    def __init__(self, mongodb: Optional[MongoDBClient] = None):
        self.rev_agent = RevisionAgent()
        self.quiz_agent = QuizAgent()
        self.feedback_agent = FeedbackAgent()
        self.qa_agent = QAAgent()
        self.conclusion_agent = ConclusionAgent()
        self.mongo = mongodb or MongoDBClient()

        # Build LangGraph flow
        self.graph = StateGraph(OrchestratorState)
        self._build_graph()

    def _build_graph(self):
        g = self.graph

        # Add nodes (no logic yet, just placeholders)
        g.add_node("EXPLAIN", lambda state: state)
        g.add_node("ASK_CHECK", lambda state: state)
        g.add_node("WAIT_ANSWER", lambda state: state)
        g.add_node("GIVE_FEEDBACK", lambda state: state)
        g.add_node("NEXT_CONCEPT", lambda state: state)

        # Add edges between them
        g.add_edge("EXPLAIN", "ASK_CHECK")
        g.add_edge("ASK_CHECK", "WAIT_ANSWER")
        g.add_edge("WAIT_ANSWER", "GIVE_FEEDBACK")
        g.add_edge("GIVE_FEEDBACK", "NEXT_CONCEPT")
        g.add_edge("NEXT_CONCEPT", "EXPLAIN")
        g.add_edge("NEXT_CONCEPT", "END")

    # utility to stringify conversation history for prompts (latest first)
    def _format_conversation_history(self, session_doc: Dict[str, Any], limit: int = 10) -> str:
        if not session_doc:
            return ""
        conv = session_doc.get("conversation_history", [])[-limit:]
        # reverse for latest-first display
        conv = list(reversed(conv))
        lines = []
        for i, turn in enumerate(conv):
            user = turn.get("user_message", "")
            assistant = turn.get("assistant_message", "")
            ts = turn.get("timestamp", "")
            lines.append(f"[{i}] user: {user} | assistant: {assistant}")
        return "\n".join(lines)

    async def start_revision_session(self, topic: str, student_id: str, session_id: str) -> Dict[str, Any]:
        """
        Create/initialize session, fetch topic content from MongoDB (subtopics), and return first explanation + interactive buttons.
        """
        # prepare or fetch session doc
        session_doc = self.mongo.get_revision_session(session_id) or {}
        if not session_doc:
            # create a new revision session doc structure
            session_doc = {
                "session_id": session_id,
                "student_id": student_id,
                "topic": topic,
                "started_at": datetime.utcnow(),
                "conversation_count": 0,
                "is_complete": False,
                "max_conversations": 999,
                "completion_threshold": 0,
                "conversation_history": [],
                # Quiz tracking
                "current_concept_correct_answers": 0,
                "required_correct_answers": 5,
                "current_concept_questions_asked": []
            }
            self.mongo.save_revision_session(session_doc)

        # fetch subtopics for the topic
        topic_title = topic.split(": ")[-1] if ": " in topic else topic
        subtopics = self.mongo.get_topic_subtopics(topic_title)
        # if not found, try get_topic_content fallback
        if not subtopics:
            subtopic_chunks = self.mongo.get_topic_content(topic)
            subtopics = [{"subtopic_number": c["id"], "subtopic_title": c.get("subtopic_title", ""), "content": c["text"]} for c in subtopic_chunks]

        # store concept chunks into session_doc
        session_doc["concept_chunks"] = subtopics
        session_doc["current_chunk_index"] = 0
        self.mongo.save_revision_session(session_doc)

        # start first concept
        return await self._present_current_concept(session_doc)

    # Update the _present_current_concept method in your orchestrator:

    async def _present_current_concept(self, session_doc: Dict[str, Any]) -> Dict[str, Any]:
        idx = session_doc.get("current_chunk_index", 0)
        chunks = session_doc.get("concept_chunks", [])
        if idx >= len(chunks):
            # Finished all concepts
            session_doc["is_complete"] = True
            self.mongo.save_revision_session(session_doc)
            summary = await self.conclusion_agent.summary(
                correct=len([p for p in session_doc.get("concepts_learned", []) if p]),
                total=len(chunks),
                conversation_history=self._format_conversation_history(session_doc)
            )
            return {"response": summary, "is_session_complete": True, "conversation_count": session_doc.get("conversation_count", 0)}
        
        current = chunks[idx]
        title = current.get("subtopic_title") or f"Concept {current.get('subtopic_number')}"
        content = current.get("content", "")
        
        # Reset quiz tracking for new concept
        session_doc["current_concept_correct_answers"] = 0
        session_doc["current_concept_questions_asked"] = []
        session_doc["has_used_learning_support"] = False
        
        # Generate structured explanation with multiple bubbles
        conv_hist = self._format_conversation_history(session_doc)
        structured_content = await self.rev_agent.generate_structured_explanation(title, content, conversation_history=conv_hist)
        
        # Add the structured content bubbles
        messages = structured_content.copy()
        
        # Add recommendation buttons message
        buttons_message = "What would you like to do next?"
        messages.append({
            "assistant_message": buttons_message, 
            "message_type": "buttons",
            "buttons": [
                {"text": "I need more examples", "action": "more_examples"},
                {"text": "Can you re-explain?", "action": "re_explain"}
                # Note: No "check understanding" button on first explanation
            ]
        })
        
        # Create separate turns for all messages
        for i, message in enumerate(messages):
            turn = {
                "turn": session_doc.get("conversation_count", 0) + 1 + i,
                "user_message": None,
                "assistant_message": message["assistant_message"],
                "stage": "explain",
                "timestamp": datetime.utcnow(),
                "concept_covered": title,
                "question_asked": False,
                "message_type": message["message_type"],
                "buttons": message.get("buttons", []),
                "section": message.get("section", "")
            }
            session_doc.setdefault("conversation_history", []).append(turn)

        # Update conversation count
        session_doc["conversation_count"] = session_doc.get("conversation_count", 0) + len(messages)
        # Set session state for button interactions
        session_doc["expecting_button_action"] = True
        session_doc["current_question_concept"] = title
        session_doc["current_content"] = content
        self.mongo.save_revision_session(session_doc)

        logger.info(f"Structured messages being returned: {len(messages)} bubbles")

        result = {
            "response": messages,
            "message_format": "multiple_bubbles",
            "is_session_complete": False, 
            "conversation_count": session_doc["conversation_count"], 
            "current_stage": "explain", 
            "current_concept": title
        }

        return result

    async def handle_user_input(self, session_id: str, user_query: str) -> Dict[str, Any]:
        session_doc = self.mongo.get_revision_session(session_id) or {}
        if not session_doc:
            return {"response":"Session not found. Start a new revision session.", "is_session_complete": True, "conversation_count": 0}

        # Get conversation history first
        conv_hist = self._format_conversation_history(session_doc)
        
        # Check if this is a button action
        if session_doc.get("expecting_button_action", False):
            return await self._handle_button_action(session_doc, user_query, conv_hist)
        
        # Check if we're expecting an answer to a check question
        if session_doc.get("expecting_answer", False):
            return await self._handle_answer_evaluation(session_doc, user_query, conv_hist)
            
        # Dynamic question detection using LLM
        current_concept = session_doc.get("current_question_concept", "")
        question_intent = await self.rev_agent.detect_question_intent(
            user_input=user_query,
            current_concept=current_concept,
            conversation_history=conv_hist
        )
        is_question = (question_intent == "ASKING_QUESTION")
        
        # Save user turn
        user_turn = {
            "turn": session_doc.get("conversation_count", 0) + 1,
            "user_message": user_query,
            "assistant_message": None,
            "stage": "user_input",
            "timestamp": datetime.utcnow(),
            "concept_covered": session_doc.get("current_question_concept")
        }
        session_doc.setdefault("conversation_history", []).append(user_turn)
        session_doc["conversation_count"] = session_doc.get("conversation_count", 0) + 1
        self.mongo.save_revision_session(session_doc)

        # Handle simple acknowledgements
        if question_intent == "ACKNOWLEDGEMENT":
            nudge = "Great! When you're ready, please choose one of the options above or ask me anything."
            assistant_turn = {
                "turn": session_doc["conversation_count"],
                "user_message": user_query,
                "assistant_message": nudge,
                "stage": "ack",
                "timestamp": datetime.utcnow(),
            }
            session_doc.setdefault("conversation_history", []).append(assistant_turn)
            self.mongo.save_revision_session(session_doc)
            return {"response": nudge, "conversation_count": session_doc["conversation_count"], "is_session_complete": False, "current_stage": "ack"}

        if is_question:
            # Handle Q&A using revision agent with current concept context
            return await self._handle_qa_request(session_doc, user_query, conv_hist)

        # Handle custom/irrelevant input
        return await self._handle_custom_input(session_doc, user_query, conv_hist)

    async def _handle_button_action(self, session_doc: Dict[str, Any], user_query: str, conv_hist: str) -> Dict[str, Any]:
        """Handle button action clicks"""
        current_chunk_idx = session_doc.get("current_chunk_index", 0)
        chunks = session_doc.get("concept_chunks", [])
        
        if current_chunk_idx >= len(chunks):
            return {"response": "No more concepts to explore.", "conversation_count": session_doc["conversation_count"], "is_session_complete": True}
            
        current = chunks[current_chunk_idx]
        title = current.get("subtopic_title") or f"Concept {current.get('subtopic_number')}"
        content = current.get("content", "")
        
        # Save user turn
        user_turn = {
            "turn": session_doc.get("conversation_count", 0) + 1,
            "user_message": user_query,
            "assistant_message": None,
            "stage": "button_action",
            "timestamp": datetime.utcnow(),
            "concept_covered": title
        }
        session_doc.setdefault("conversation_history", []).append(user_turn)
        session_doc["conversation_count"] = session_doc.get("conversation_count", 0) + 1
        
        # Handle mastery buttons (after concept completion)
        if session_doc.get("concept_mastered", False):
            if "more questions" in user_query.lower() or user_query.lower().strip() == "more_questions":
                # Continue with more questions for current concept
                session_doc["concept_mastered"] = False  # Reset mastery flag
                session_doc["expecting_button_action"] = False
                session_doc["expecting_answer"] = True
                
                # Generate new question
                next_question = await self.rev_agent.make_check_question(title, content, conversation_history=conv_hist)
                session_doc.setdefault("current_concept_questions_asked", []).append(next_question)
                
                correct_answers = session_doc.get("current_concept_correct_answers", 0)
                response_message = f"Great! Let's continue with more questions to deepen your understanding.\n\n**Additional Question {correct_answers + 1}:**\n{next_question}"
                
                # Set up for answer evaluation
                try:
                    expected_keywords = await self.rev_agent.extract_expected_keywords(title, content, next_question)
                except Exception:
                    expected_keywords = [w for w in (title.split()[:3])]
                
                session_doc["current_expected_keywords"] = expected_keywords
                session_doc["current_question"] = next_question
                
                # Save assistant turn
                turn = {
                    "turn": session_doc.get("conversation_count", 0) + 1,
                    "user_message": None,
                    "assistant_message": response_message,
                    "stage": "additional_question",
                    "timestamp": datetime.utcnow(),
                    "concept_covered": title,
                    "message_type": "question",
                    "is_additional_question": True
                }
                session_doc.setdefault("conversation_history", []).append(turn)
                session_doc["conversation_count"] = session_doc.get("conversation_count", 0) + 1
                
                self.mongo.save_revision_session(session_doc)
                
                return {
                    "response": response_message,
                    "message_format": "single",
                    "conversation_count": session_doc["conversation_count"],
                    "is_session_complete": False,
                    "current_stage": "additional_question"
                }
                
            elif "next concept" in user_query.lower() or user_query.lower().strip() == "next_concept":
                # Move to next concept
                session_doc["concept_mastered"] = False  # Reset mastery flag
                session_doc["current_chunk_index"] = session_doc.get("current_chunk_index", 0) + 1
                session_doc["expecting_answer"] = False
                session_doc["expecting_button_action"] = False
                session_doc["current_question_concept"] = None
                
                # Present next concept
                next_payload = await self._present_current_concept(session_doc)
                
                # Create transition message
                transition_message = "Perfect! Moving to the next concept..."
                messages = [{"assistant_message": transition_message, "message_type": "transition"}]
                
                next_response = next_payload.get("response", [])
                if isinstance(next_response, list):
                    messages.extend(next_response)
                elif next_response:
                    messages.append({"assistant_message": next_response, "message_type": "concept"})
                
                # Save transition turn
                turn = {
                    "turn": session_doc.get("conversation_count", 0) + 1,
                    "user_message": None,
                    "assistant_message": transition_message,
                    "stage": "concept_transition",
                    "timestamp": datetime.utcnow(),
                    "message_type": "transition"
                }
                session_doc.setdefault("conversation_history", []).append(turn)
                session_doc["conversation_count"] = session_doc.get("conversation_count", 0) + 1
                
                self.mongo.save_revision_session(session_doc)
                
                return {
                    "response": messages,
                    "message_format": "multiple_bubbles",
                    "conversation_count": session_doc["conversation_count"],
                    "is_session_complete": next_payload.get("is_session_complete", False),
                    "current_stage": "concept_transition"
                }
        
        # Handle regular learning buttons (during concept learning)
        response_message = ""
        next_action = "continue"  # continue, check_understanding, next_concept
        
        if "more examples" in user_query.lower() or user_query.lower().strip() == "more_examples":
            # Generate more examples
            response_message = await self.rev_agent.generate_examples(title, content, conversation_history=conv_hist)
            next_action = "continue"
            # Mark that student has engaged with learning support
            session_doc["has_used_learning_support"] = True
            
        elif "re-explain" in user_query.lower() or user_query.lower().strip() == "re_explain":
            # Re-explain the concept
            steps = await self.rev_agent.generate_explanation_steps(title, content, conversation_history=conv_hist, steps=4)
            response_message = "Let me explain this concept again in a different way:\n\n" + "\n".join(steps)
            next_action = "continue"
            # Mark that student has engaged with learning support
            session_doc["has_used_learning_support"] = True
            
        elif "check my understanding" in user_query.lower() or user_query.lower().strip() == "check_understanding":
            # Start quiz mode - generate first question
            return await self._start_quiz_mode(session_doc, title, content, conv_hist)
            
        else:
    # Check if this might be a question rather than a button action
            current_concept = session_doc.get("current_question_concept", "")
            current_content = session_doc.get("current_content", "")
            
            # Check relevance of the input
            try:
                relevance = await self.rev_agent.check_question_relevance(user_query, current_concept, current_content)
                
                if relevance == "RELEVANT":
                    # Handle as relevant question
                    response_message = await self.rev_agent.handle_qa_request(
                        user_question=user_query,
                        current_concept=current_concept,
                        content=current_content,
                        conversation_history=conv_hist
                    )
                else:
                    # Handle as irrelevant input with polite redirect
                    response_message = await self.rev_agent.handle_custom_input(
                        user_input=user_query,
                        current_concept=current_concept,
                        content=current_content,
                        conversation_history=conv_hist
                    )
            except Exception as e:
                # Fallback for errors
                response_message = "I didn't understand that action. Please choose one of the available options or ask me a question related to our current topic."
            
            next_action = "continue"
        # Create assistant response
        messages = [{"assistant_message": response_message, "message_type": "response"}]
        
        # Add interactive buttons if continuing with current concept
        if next_action == "continue":
            buttons_message = "What would you like to do next?"
            
            # Determine which buttons to show based on learning engagement
            has_used_support = session_doc.get("has_used_learning_support", False)
            
            if has_used_support:
                # Show all buttons including Q&A after using learning support
                buttons = [
                    {"text": "I need more examples", "action": "more_examples"},
                    {"text": "Can you re-explain?", "action": "re_explain"},
                    {"text": "Let me check my understanding with some Q&A", "action": "check_understanding"}
                ]
            else:
                # Show only learning support buttons initially
                buttons = [
                    {"text": "I need more examples", "action": "more_examples"},
                    {"text": "Can you re-explain?", "action": "re_explain"}
                ]
            
            messages.append({
                "assistant_message": buttons_message,
                "message_type": "buttons",
                "buttons": buttons
            })
            
        # Save assistant turns
        for i, message in enumerate(messages):
            turn = {
                "turn": session_doc.get("conversation_count", 0) + 1 + i,
                "user_message": None,
                "assistant_message": message["assistant_message"],
                "stage": "button_response",
                "timestamp": datetime.utcnow(),
                "concept_covered": title,
                "message_type": message["message_type"],
                "buttons": message.get("buttons", [])
            }
            session_doc.setdefault("conversation_history", []).append(turn)
        
        session_doc["conversation_count"] = session_doc.get("conversation_count", 0) + len(messages)
        self.mongo.save_revision_session(session_doc)
        
        return {
            "response": messages,
            "message_format": "multiple_bubbles",
            "conversation_count": session_doc["conversation_count"],
            "is_session_complete": False,
            "current_stage": "button_response"
        }

    async def _start_quiz_mode(self, session_doc: Dict[str, Any], title: str, content: str, conv_hist: str) -> Dict[str, Any]:
        """Start quiz mode for the current concept"""
        # Generate first question
        check_q = await self.rev_agent.make_check_question(title, content, conversation_history=conv_hist)
        
        # Track this question
        session_doc.setdefault("current_concept_questions_asked", []).append(check_q)
        
        correct_so_far = session_doc.get("current_concept_correct_answers", 0)
        required_total = session_doc.get("required_correct_answers", 5)
        
        response_message = f"Great! Let's test your understanding. You need to answer {required_total} questions correctly to master this concept.\n\n**Progress: {correct_so_far}/{required_total} correct answers**\n\n**Question {correct_so_far + 1}:**\n{check_q}"
        
        # Set up for answer evaluation
        try:
            expected_keywords = await self.rev_agent.extract_expected_keywords(title, content, check_q)
        except Exception:
            expected_keywords = [w for w in (title.split()[:3])]
        
        session_doc["expecting_answer"] = True
        session_doc["current_expected_keywords"] = expected_keywords
        session_doc["expecting_button_action"] = False
        session_doc["current_question"] = check_q
        
        # Save assistant turn
        turn = {
            "turn": session_doc.get("conversation_count", 0) + 1,
            "user_message": None,
            "assistant_message": response_message,
            "stage": "quiz_question",
            "timestamp": datetime.utcnow(),
            "concept_covered": title,
            "message_type": "question",
            "question_number": correct_so_far + 1,
            "progress": f"{correct_so_far}/{required_total}"
        }
        session_doc.setdefault("conversation_history", []).append(turn)
        session_doc["conversation_count"] = session_doc.get("conversation_count", 0) + 1
        
        self.mongo.save_revision_session(session_doc)
        
        return {
            "response": response_message,
            "message_format": "single",
            "conversation_count": session_doc["conversation_count"],
            "is_session_complete": False,
            "current_stage": "quiz_question"
        }

    async def _handle_qa_request(self, session_doc: Dict[str, Any], user_query: str, conv_hist: str) -> Dict[str, Any]:
        """Handle Q&A requests"""
        current_concept = session_doc.get("current_question_concept", "")
        current_chunk_idx = session_doc.get("current_chunk_index", 0)
        concept_chunks = session_doc.get("concept_chunks", [])
        current_content = ""
        
        if current_chunk_idx < len(concept_chunks):
            current_content = concept_chunks[current_chunk_idx].get("content", "")
        
        # First check relevance
        relevance = await self.rev_agent.check_question_relevance(user_query, current_concept, current_content)
        
        if relevance == "RELEVANT":
            # Answer the relevant question
            answer = await self.rev_agent.handle_qa_request(
                user_question=user_query,
                current_concept=current_concept,
                content=current_content,
                conversation_history=conv_hist
            )
            combined = answer.strip()
        else:
            # Handle irrelevant question with polite redirect
            combined = await self.rev_agent.handle_custom_input(
                user_input=user_query,
                current_concept=current_concept,
                content=current_content,
                conversation_history=conv_hist
            )

        # Create response with continue buttons
        buttons_message = "What would you like to do next?"

        messages = [
            {"assistant_message": combined, "message_type": "qa_response"},
            {
                "assistant_message": buttons_message,
                "message_type": "buttons",
                "buttons": [
                    {"text": "I need more examples", "action": "more_examples"},
                    {"text": "Can you re-explain?", "action": "re_explain"},
                    {"text": "Let me check my understanding with some Q&A", "action": "check_understanding"}
                ]
            }
        ]

        # Save assistant turns
        for i, message in enumerate(messages):
            turn = {
                "turn": session_doc.get("conversation_count", 0) + 1 + i,
                "user_message": None,
                "assistant_message": message["assistant_message"],
                "stage": "qa",
                "timestamp": datetime.utcnow(),
                "message_type": message["message_type"],
                "buttons": message.get("buttons", [])
            }
            session_doc.setdefault("conversation_history", []).append(turn)
        
        session_doc["conversation_count"] = session_doc.get("conversation_count", 0) + len(messages)
        session_doc["expecting_button_action"] = True
        self.mongo.save_revision_session(session_doc)

        return {
            "response": messages,
            "message_format": "multiple_bubbles",
            "conversation_count": session_doc["conversation_count"],
            "is_session_complete": False,
            "current_stage": "qa"
        }

    async def _handle_custom_input(self, session_doc: Dict[str, Any], user_query: str, conv_hist: str) -> Dict[str, Any]:
        """Handle custom user input that doesn't fit standard categories"""
        current_concept = session_doc.get("current_question_concept", "")
        current_chunk_idx = session_doc.get("current_chunk_index", 0)
        concept_chunks = session_doc.get("concept_chunks", [])
        current_content = ""
        
        if current_chunk_idx < len(concept_chunks):
            current_content = concept_chunks[current_chunk_idx].get("content", "")
        
        # Use revision agent to handle custom input contextually
        response = await self.rev_agent.handle_custom_input(
            user_input=user_query,
            current_concept=current_concept,
            content=current_content,
            conversation_history=conv_hist
        )
        
        # Create response with appropriate buttons
        buttons_message = "What would you like to do next?"
        
        # Determine which buttons to show based on learning engagement
        has_used_support = session_doc.get("has_used_learning_support", False)
        
        if has_used_support:
            # Show all buttons including Q&A after using learning support
            buttons = [
                {"text": "I need more examples", "action": "more_examples"},
                {"text": "Can you re-explain?", "action": "re_explain"},
                {"text": "Let me check my understanding with some Q&A", "action": "check_understanding"}
            ]
        else:
            # Show only learning support buttons initially
            buttons = [
                {"text": "I need more examples", "action": "more_examples"},
                {"text": "Can you re-explain?", "action": "re_explain"}
            ]

        messages = [
            {"assistant_message": response, "message_type": "custom_response"},
            {
                "assistant_message": buttons_message,
                "message_type": "buttons",
                "buttons": buttons
            }
        ]

        # Save assistant turns
        for i, message in enumerate(messages):
            turn = {
                "turn": session_doc.get("conversation_count", 0) + 1 + i,
                "user_message": None,
                "assistant_message": message["assistant_message"],
                "stage": "custom_input",
                "timestamp": datetime.utcnow(),
                "message_type": message["message_type"],
                "buttons": message.get("buttons", [])
            }
            session_doc.setdefault("conversation_history", []).append(turn)
        
        session_doc["conversation_count"] = session_doc.get("conversation_count", 0) + len(messages)
        session_doc["expecting_button_action"] = True
        self.mongo.save_revision_session(session_doc)

        return {
            "response": messages,
            "message_format": "multiple_bubbles",
            "conversation_count": session_doc["conversation_count"],
            "is_session_complete": False,
            "current_stage": "custom_input"
        }

    async def _handle_answer_evaluation(self, session_doc: Dict[str, Any], user_query: str, conv_hist: str) -> Dict[str, Any]:
        """Handle answer evaluation for quiz questions"""
        expected_keywords = session_doc.get("current_expected_keywords", [])
        
        # Build full context for evaluation
        current_chunk_idx = session_doc.get("current_chunk_index", 0)
        chunks = session_doc.get("concept_chunks", [])
        title = session_doc.get("current_question_concept", "")
        content = ""
        if current_chunk_idx < len(chunks):
            content = chunks[current_chunk_idx].get("content", "")
            
        # Get the current question
        check_question = session_doc.get("current_question", "")
                
        eval_result = await self.rev_agent.evaluate_answer(
            user_query,
            expected_keywords,
            conversation_history=conv_hist,
            title=title,
            content=content,
            assistant_message="",
            check_question=check_question
        )
        
        verdict = eval_result.get("verdict", "WRONG")
        correct_answers = session_doc.get("current_concept_correct_answers", 0)
        required_total = session_doc.get("required_correct_answers", 5)
        
        if verdict == "CORRECT":
            # Check if this is an additional question (after mastery)
            if session_doc.get("concept_mastered", False) and session_doc.get("current_concept_correct_answers", 0) >= required_total:
                # This is an additional question after mastery
                assistant_feedback = f"CORRECT!\nExcellent! You continue to demonstrate strong understanding of this concept.\n\nWhat you got right:\n{eval_result.get('justification', 'Great explanation!')}"
                
                # Offer more questions or move to next concept
                recommendation_message = "What would you like to do next?"
                
                messages = [
                    {"assistant_message": assistant_feedback, "message_type": "additional_correct"},
                    {
                        "assistant_message": recommendation_message,
                        "message_type": "mastery_buttons",
                        "buttons": [
                            {"text": "Could you provide a few more questions?", "action": "more_questions"},
                            {"text": "Can you move to the next concept?", "action": "next_concept"}
                        ]
                    }
                ]
                
                # Save feedback turns
                for i, message in enumerate(messages):
                    turn = {
                        "turn": session_doc.get("conversation_count", 0) + 1 + i,
                        "user_message": user_query if i == 0 else None,
                        "assistant_message": message["assistant_message"],
                        "stage": "additional_correct",
                        "timestamp": datetime.utcnow(),
                        "correct_answer": True,
                        "is_additional_correct": True,
                        "message_type": message["message_type"],
                        "buttons": message.get("buttons", [])
                    }
                    session_doc.setdefault("conversation_history", []).append(turn)
                
                session_doc["conversation_count"] = session_doc.get("conversation_count", 0) + len(messages)
                session_doc["expecting_answer"] = False
                session_doc["expecting_button_action"] = True
                
                self.mongo.save_revision_session(session_doc)
                
                return {
                    "response": messages,
                    "message_format": "multiple_bubbles",
                    "conversation_count": session_doc["conversation_count"],
                    "is_session_complete": False,
                    "current_stage": "additional_correct"
                }
            
            # Regular progression (not yet mastered)
            # Increment correct answers
            correct_answers += 1
            session_doc["current_concept_correct_answers"] = correct_answers
            
            # Provide positive feedback
            assistant_feedback = f"CORRECT!\nGreat job! Your answer is absolutely right. You covered all the key points:\n\n**Progress: {correct_answers}/{required_total} correct answers**"
            
            # Check if concept is mastered (5 correct answers)
            if correct_answers >= required_total:
                # Concept completed - show recommendation options
                session_doc.setdefault("concepts_learned", []).append(title)
                session_doc["expecting_answer"] = False
                session_doc["expecting_button_action"] = True
                session_doc["concept_mastered"] = True  # Flag to track mastery state
                
                # Add completion message with recommendations
                completion_message = f"\n\n🎉 **Concept Mastered!**\nYou've successfully answered {required_total} questions correctly."
                assistant_feedback += completion_message
                
                # Create recommendation message
                recommendation_message = "What would you like to do next?"
                
                messages = [
                    {"assistant_message": assistant_feedback, "message_type": "mastery_feedback"},
                    {
                        "assistant_message": recommendation_message,
                        "message_type": "mastery_buttons",
                        "buttons": [
                            {"text": "Could you provide a few more questions?", "action": "more_questions"},
                            {"text": "Can you move to the next concept?", "action": "next_concept"}
                        ]
                    }
                ]
                
                # Save feedback turns
                for i, message in enumerate(messages):
                    turn = {
                        "turn": session_doc.get("conversation_count", 0) + 1 + i,
                        "user_message": user_query if i == 0 else None,
                        "assistant_message": message["assistant_message"],
                        "stage": "concept_mastered",
                        "timestamp": datetime.utcnow(),
                        "correct_answer": True,
                        "concept_mastered": True,
                        "message_type": message["message_type"],
                        "buttons": message.get("buttons", [])
                    }
                    session_doc.setdefault("conversation_history", []).append(turn)
                
                session_doc["conversation_count"] = session_doc.get("conversation_count", 0) + len(messages)
                self.mongo.save_revision_session(session_doc)
                
                return {
                    "response": messages,
                    "message_format": "multiple_bubbles",
                    "conversation_count": session_doc["conversation_count"],
                    "is_session_complete": False,
                    "current_stage": "concept_mastered"
                }
            else:
                # Ask next question
                next_question = await self.rev_agent.make_check_question(title, content, conversation_history=conv_hist)
                session_doc.setdefault("current_concept_questions_asked", []).append(next_question)
                
                next_question_message = f"\nLet's try another question:\n\n**Question {correct_answers + 1}:**\n{next_question}"
                combined_message = assistant_feedback + next_question_message
                
                # Update session for next question
                try:
                    expected_keywords = await self.rev_agent.extract_expected_keywords(title, content, next_question)
                except Exception:
                    expected_keywords = [w for w in (title.split()[:3])]
                    
                session_doc["current_expected_keywords"] = expected_keywords
                session_doc["current_question"] = next_question
                
                # Save turns
                feedback_turn = {
                    "turn": session_doc.get("conversation_count", 0) + 1,
                    "user_message": user_query,
                    "assistant_message": combined_message,
                    "stage": "correct_answer_next_question",
                    "timestamp": datetime.utcnow(),
                    "correct_answer": True,
                    "question_number": correct_answers + 1,
                    "progress": f"{correct_answers}/{required_total}"
                }
                session_doc.setdefault("conversation_history", []).append(feedback_turn)
                session_doc["conversation_count"] = session_doc.get("conversation_count", 0) + 1
                
                self.mongo.save_revision_session(session_doc)
                
                return {
                    "response": combined_message,
                    "message_format": "single",
                    "conversation_count": session_doc["conversation_count"],
                    "is_session_complete": False,
                    "current_stage": "next_question"
                }
        else:
            # Wrong/partial answer - provide feedback with retry buttons
            assistant_feedback = self.feedback_agent.feedback_for(verdict, {"correction": eval_result.get("correction", eval_result.get("justification", ""))})
            session_doc["expecting_answer"] = False
            session_doc["expecting_button_action"] = True
            
            feedback_message = assistant_feedback
            buttons_message = "Let's try a different approach. What would you like to do?"

            messages = [
                {"assistant_message": feedback_message, "message_type": "feedback"},
                {
                    "assistant_message": buttons_message,
                    "message_type": "buttons", 
                    "buttons": [
                        {"text": "I need more examples", "action": "more_examples"},
                        {"text": "Can you re-explain?", "action": "re_explain"},
                        {"text": "Let me check my understanding with some Q&A", "action": "check_understanding"}
                    ]
                }
            ]
            
            # Save turns
            for i, message in enumerate(messages):
                turn = {
                    "turn": session_doc.get("conversation_count", 0) + 1 + i,
                    "user_message": user_query if i == 0 else None,
                    "assistant_message": message["assistant_message"],
                    "stage": "wrong_answer_feedback",
                    "timestamp": datetime.utcnow(),
                    "correct_answer": False,
                    "message_type": message["message_type"],
                    "buttons": message.get("buttons", [])
                }
                session_doc.setdefault("conversation_history", []).append(turn)
            
            session_doc["conversation_count"] = session_doc.get("conversation_count", 0) + len(messages)
            self.mongo.save_revision_session(session_doc)
            
            return {
                "response": messages,
                "message_format": "multiple_bubbles", 
                "conversation_count": session_doc["conversation_count"],
                "is_session_complete": False,
                "current_stage": "wrong_answer_feedback"
            }