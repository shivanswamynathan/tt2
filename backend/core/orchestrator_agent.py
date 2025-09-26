from typing import TypedDict, List, Dict, Any, Optional
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
from langgraph.graph import StateGraph, END

class OrchestratorState(TypedDict):
    session_id: str
    student_id: str
    topic: str
    started_at: Optional[datetime]
    conversation_count: int
    is_complete: bool
    max_conversations: int
    completion_threshold: int
    conversation_history: List[Dict[str, Any]]
    current_concept_correct_answers: int
    required_correct_answers: int
    current_concept_questions_asked: List[str]
    concept_chunks: List[Dict[str, Any]]
    current_chunk_index: int
    has_used_learning_support: bool
    expecting_button_action: bool
    current_question_concept: Optional[str]
    current_content: str
    expecting_answer: bool
    current_expected_keywords: Optional[List[str]]
    current_question: Optional[str]
    concept_mastered: bool
    concepts_learned: List[str]
    user_message: Optional[str]
    assistant_message: Optional[str]
    stage: Optional[str]
    intent: Optional[str]
    response: Any
    message_format: Optional[str]
    is_session_complete: bool
    current_stage: str
    next_action: Optional[str]  # Temporary flag for routing within graph

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
        self.app = self.graph.compile()

    def _build_graph(self):
        g = self.graph

        # Add nodes
        g.add_node("handle_input", self.handle_input_node)
        g.add_node("detect_intent", self.detect_intent_node)
        g.add_node("handle_ack", self.handle_ack_node)
        g.add_node("handle_qa", self.handle_qa_node)
        g.add_node("handle_custom", self.handle_custom_node)
        g.add_node("handle_button", self.handle_button_node)
        g.add_node("evaluate_answer", self.evaluate_answer_node)
        g.add_node("present_concept", self.present_concept_node)
        g.add_node("conclusion", self.conclusion_node)

        # Set conditional entry point
        g.set_conditional_entry_point(
            self.entry_route,
            path_map={
                "present_concept": "present_concept",
                "handle_input": "handle_input",
            },
        )

        # Edges after handle_input
        g.add_conditional_edges(
            "handle_input",
            self.route_after_input,
            {
                "handle_button": "handle_button",
                "evaluate_answer": "evaluate_answer",
                "detect_intent": "detect_intent",
                "conclusion": "conclusion",
            },
        )

        # Edges after detect_intent
        g.add_conditional_edges(
            "detect_intent",
            self.route_after_detect,
            {
                "handle_ack": "handle_ack",
                "handle_qa": "handle_qa",
                "handle_custom": "handle_custom",
            },
        )

        # Edges after handle_button
        g.add_conditional_edges(
            "handle_button",
            self.route_after_button,
            {
                "present_concept": "present_concept",
                "end": END,
            },
        )

        # Direct edges to END
        g.add_edge("handle_ack", END)
        g.add_edge("handle_qa", END)
        g.add_edge("handle_custom", END)
        g.add_edge("evaluate_answer", END)
        g.add_edge("conclusion", END)
        g.add_edge("present_concept", END)

    def entry_route(self, state: OrchestratorState) -> str:
        if state.get("user_message") is None:
            return "present_concept"
        else:
            return "handle_input"

    def route_after_input(self, state: OrchestratorState) -> str:
        if state["is_complete"]:
            return "conclusion"
        if state.get("expecting_button_action", False):
            return "handle_button"
        if state.get("expecting_answer", False):
            return "evaluate_answer"
        return "detect_intent"

    def route_after_detect(self, state: OrchestratorState) -> str:
        intent = state.get("intent")
        if intent == "ACKNOWLEDGEMENT":
            return "handle_ack"
        elif intent == "ASKING_QUESTION":
            return "handle_qa"
        else:
            return "handle_custom"

    def route_after_button(self, state: OrchestratorState) -> str:
        if state.get("next_action") == "present_concept":
            return "present_concept"
        return "end"

    # utility to stringify conversation history for prompts (latest first)
    def _format_conversation_history(self, state: Dict[str, Any], limit: int = 10) -> str:
        if not state:
            return ""
        conv = state.get("conversation_history", [])[-limit:]
        # reverse for latest-first display
        conv = list(reversed(conv))
        lines = []
        for i, turn in enumerate(conv):
            user = turn.get("user_message", "")
            assistant = turn.get("assistant_message", "")
            ts = turn.get("timestamp", "")
            lines.append(f"[{i}] user: {user} | assistant: {assistant}")
        return "\n".join(lines)

    def start_revision_session(self, topic: str, student_id: str, session_id: str) -> Dict[str, Any]:
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
                "current_concept_questions_asked": [],
                "concept_chunks": [],
                "current_chunk_index": 0,
                "has_used_learning_support": False,
                "expecting_button_action": False,
                "current_question_concept": None,
                "current_content": "",
                "expecting_answer": False,
                "current_expected_keywords": None,
                "current_question": None,
                "concept_mastered": False,
                "concepts_learned": [],
                "is_session_complete": False,
                "current_stage": ""
            }

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

        # Invoke graph for present_concept
        state = OrchestratorState(**session_doc)
        output_state = self.app.invoke(state)

        # Save updated state to mongo
        self.mongo.save_revision_session(output_state)

        # Extract result
        result = {
            "response": output_state["response"],
            "message_format": output_state.get("message_format"),
            "is_session_complete": output_state["is_session_complete"],
            "conversation_count": output_state["conversation_count"],
            "current_stage": output_state["current_stage"],
            "current_concept": output_state.get("current_question_concept")
        }
        return result

    def handle_user_input(self, session_id: str, user_query: str) -> Dict[str, Any]:
        session_doc = self.mongo.get_revision_session(session_id) or {}
        if not session_doc:
            return {"response":"Session not found. Start a new revision session.", "is_session_complete": True, "conversation_count": 0}

        state = OrchestratorState(**session_doc)
        state["user_message"] = user_query

        output_state = self.app.invoke(state)

        # Save updated state to mongo
        self.mongo.save_revision_session(output_state)

        # Extract result
        result = {
            "response": output_state["response"],
            "message_format": output_state.get("message_format"),
            "is_session_complete": output_state["is_session_complete"],
            "conversation_count": output_state["conversation_count"],
            "current_stage": output_state["current_stage"]
        }
        return result

    def present_concept_node(self, state: OrchestratorState) -> Dict[str, Any]:
        idx = state.get("current_chunk_index", 0)
        chunks = state.get("concept_chunks", [])
        if idx >= len(chunks):
            # Finished all concepts
            conv_hist = self._format_conversation_history(state)
            summary = self.conclusion_agent.summary.invoke({
                "correct": len(state.get("concepts_learned", [])),
                "total": len(chunks),
                "conversation_history": conv_hist
            })
            updates = {
                "is_complete": True,
                "response": summary,
                "is_session_complete": True,
                "conversation_count": state.get("conversation_count", 0)
            }
            return updates
        
        current = chunks[idx]
        title = current.get("subtopic_title") or f"Concept {current.get('subtopic_number')}"
        content = current.get("content", "")
        
        # Reset quiz tracking for new concept
        state["current_concept_correct_answers"] = 0
        state["current_concept_questions_asked"] = []
        state["has_used_learning_support"] = False
        
        # Generate structured explanation with multiple bubbles
        conv_hist = self._format_conversation_history(state)
        structured_content = self.rev_agent.generate_structured_explanation.invoke({
            "title": title, 
            "content": content, 
            "conversation_history": conv_hist
        })
        
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
        
        # If there's an existing response (e.g., transition), append
        existing_response = state.get("response", [])
        if not isinstance(existing_response, list):
            existing_response = []
        existing_response.extend(messages)
        messages = existing_response
        
        # Create separate turns for all messages
        for i, message in enumerate(messages):
            turn = {
                "turn": state.get("conversation_count", 0) + 1 + i,
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
            state["conversation_history"].append(turn)

        # Update conversation count
        state["conversation_count"] = state.get("conversation_count", 0) + len(messages)
        # Set session state for button interactions
        state["expecting_button_action"] = True
        state["current_question_concept"] = title
        state["current_content"] = content

        logger.info(f"Structured messages being returned: {len(messages)} bubbles")

        updates = {
            "conversation_history": state["conversation_history"],
            "conversation_count": state["conversation_count"],
            "expecting_button_action": state["expecting_button_action"],
            "current_question_concept": state["current_question_concept"],
            "current_content": state["current_content"],
            "response": messages,
            "message_format": "multiple_bubbles",
            "is_session_complete": False, 
            "current_stage": "explain"
        }
        return updates

    def handle_input_node(self, state: OrchestratorState) -> Dict[str, Any]:
        # Save user turn
        user_turn = {
            "turn": state.get("conversation_count", 0) + 1,
            "user_message": state["user_message"],
            "assistant_message": None,
            "stage": "user_input",
            "timestamp": datetime.utcnow(),
            "concept_covered": state.get("current_question_concept")
        }
        state["conversation_history"].append(user_turn)
        state["conversation_count"] = state.get("conversation_count", 0) + 1

        return {
            "conversation_history": state["conversation_history"],
            "conversation_count": state["conversation_count"]
        }

    def detect_intent_node(self, state: OrchestratorState) -> Dict[str, Any]:
        conv_hist = self._format_conversation_history(state)
        current_concept = state.get("current_question_concept", "")
        question_intent = self.rev_agent.detect_question_intent.invoke({
            "user_input": state["user_message"],
            "current_concept": current_concept,
            "conversation_history": conv_hist
        })
        return {"intent": question_intent}

    def handle_ack_node(self, state: OrchestratorState) -> Dict[str, Any]:
        nudge = "Great! When you're ready, please choose one of the options above or ask me anything."
        assistant_turn = {
            "turn": state["conversation_count"] + 1,
            "user_message": None,
            "assistant_message": nudge,
            "stage": "ack",
            "timestamp": datetime.utcnow(),
        }
        state["conversation_history"].append(assistant_turn)
        state["conversation_count"] += 1

        return {
            "conversation_history": state["conversation_history"],
            "conversation_count": state["conversation_count"],
            "response": nudge,
            "is_session_complete": False,
            "current_stage": "ack"
        }

    def handle_button_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Handle button action clicks"""
        current_chunk_idx = state.get("current_chunk_index", 0)
        chunks = state.get("concept_chunks", [])
        
        if current_chunk_idx >= len(chunks):
            return {
                "response": "No more concepts to explore.",
                "conversation_count": state["conversation_count"],
                "is_session_complete": True
            }
            
        current = chunks[current_chunk_idx]
        title = current.get("subtopic_title") or f"Concept {current.get('subtopic_number')}"
        content = current.get("content", "")
        
        # Save user turn (already saved in handle_input, but ensure consistency)
        user_query = state["user_message"]
        
        # Handle mastery buttons (after concept completion)
        if state.get("concept_mastered", False):
            if "more questions" in user_query.lower() or user_query.lower().strip() == "more_questions":
                # Continue with more questions for current concept
                state["concept_mastered"] = False  # Reset mastery flag
                state["expecting_button_action"] = False
                state["expecting_answer"] = True
                
                conv_hist = self._format_conversation_history(state)
                # Generate new question
                next_question = self.rev_agent.make_check_question.invoke({
                    "title": title,
                    "content": content,
                    "conversation_history": conv_hist
                })
                state["current_concept_questions_asked"].append(next_question)
                
                correct_answers = state.get("current_concept_correct_answers", 0)
                response_message = f"Great! Let's continue with more questions to deepen your understanding.\n\n**Additional Question {correct_answers + 1}:**\n{next_question}"
                
                # Set up for answer evaluation
                try:
                    expected_keywords = self.rev_agent.extract_expected_keywords.invoke({
                        "title": title,
                        "content": content,
                        "question": next_question
                    })
                except Exception:
                    expected_keywords = [w for w in (title.split()[:3])]
                
                state["current_expected_keywords"] = expected_keywords
                state["current_question"] = next_question
                
                # Save assistant turn
                turn = {
                    "turn": state.get("conversation_count", 0) + 1,
                    "user_message": None,
                    "assistant_message": response_message,
                    "stage": "additional_question",
                    "timestamp": datetime.utcnow(),
                    "concept_covered": title,
                    "message_type": "question",
                    "is_additional_question": True
                }
                state["conversation_history"].append(turn)
                state["conversation_count"] += 1
                
                return {
                    "concept_mastered": state["concept_mastered"],
                    "expecting_button_action": state["expecting_button_action"],
                    "expecting_answer": state["expecting_answer"],
                    "current_concept_questions_asked": state["current_concept_questions_asked"],
                    "current_expected_keywords": state["current_expected_keywords"],
                    "current_question": state["current_question"],
                    "conversation_history": state["conversation_history"],
                    "conversation_count": state["conversation_count"],
                    "response": response_message,
                    "message_format": "single",
                    "is_session_complete": False,
                    "current_stage": "additional_question"
                }
                
            elif "next concept" in user_query.lower() or user_query.lower().strip() == "next_concept":
                # Move to next concept
                state["concept_mastered"] = False  # Reset mastery flag
                state["current_chunk_index"] += 1
                state["expecting_answer"] = False
                state["expecting_button_action"] = False
                state["current_question_concept"] = None
                
                # Set transition message and flag for present_concept
                transition_message = "Perfect! Moving to the next concept..."
                state["response"] = [{"assistant_message": transition_message, "message_type": "transition"}]
                state["message_format"] = "multiple_bubbles"
                
                # Save transition turn
                turn = {
                    "turn": state.get("conversation_count", 0) + 1,
                    "user_message": None,
                    "assistant_message": transition_message,
                    "stage": "concept_transition",
                    "timestamp": datetime.utcnow(),
                    "message_type": "transition"
                }
                state["conversation_history"].append(turn)
                state["conversation_count"] += 1
                
                # Set flag for routing to present_concept
                state["next_action"] = "present_concept"
                
                return {
                    "concept_mastered": state["concept_mastered"],
                    "current_chunk_index": state["current_chunk_index"],
                    "expecting_answer": state["expecting_answer"],
                    "expecting_button_action": state["expecting_button_action"],
                    "current_question_concept": state["current_question_concept"],
                    "response": state["response"],
                    "message_format": state["message_format"],
                    "conversation_history": state["conversation_history"],
                    "conversation_count": state["conversation_count"],
                    "next_action": state["next_action"],
                    "is_session_complete": False,
                    "current_stage": "concept_transition"
                }
        
        # Handle regular learning buttons (during concept learning)
        conv_hist = self._format_conversation_history(state)
        response_message = ""
        next_action = "continue"  # continue, check_understanding, next_concept
        
        if "more examples" in user_query.lower() or user_query.lower().strip() == "more_examples":
            # Generate more examples
            response_message = self.rev_agent.generate_examples.invoke({
                "title": title,
                "content": content,
                "conversation_history": conv_hist
            })
            next_action = "continue"
            # Mark that student has engaged with learning support
            state["has_used_learning_support"] = True
            
        elif "re-explain" in user_query.lower() or user_query.lower().strip() == "re_explain":
            # Re-explain the concept
            steps = self.rev_agent.generate_explanation_steps.invoke({
                "title": title,
                "content": content,
                "conversation_history": conv_hist,
                "steps": 4
            })
            response_message = "Let me explain this concept again in a different way:\n\n" + "\n".join(steps)
            next_action = "continue"
            # Mark that student has engaged with learning support
            state["has_used_learning_support"] = True
            
        elif "check my understanding" in user_query.lower() or user_query.lower().strip() == "check_understanding":
            # Start quiz mode
            check_q = self.rev_agent.make_check_question.invoke({
                "title": title,
                "content": content,
                "conversation_history": conv_hist
            })
            
            # Track this question
            state["current_concept_questions_asked"].append(check_q)
            
            correct_so_far = state.get("current_concept_correct_answers", 0)
            required_total = state.get("required_correct_answers", 5)
            
            response_message = f"Great! Let's test your understanding. You need to answer {required_total} questions correctly to master this concept.\n\n**Progress: {correct_so_far}/{required_total} correct answers**\n\n**Question {correct_so_far + 1}:**\n{check_q}"
            
            # Set up for answer evaluation
            try:
                expected_keywords = self.rev_agent.extract_expected_keywords.invoke({
                    "title": title,
                    "content": content,
                    "question": check_q
                })
            except Exception:
                expected_keywords = [w for w in (title.split()[:3])]
            
            state["expecting_answer"] = True
            state["current_expected_keywords"] = expected_keywords
            state["expecting_button_action"] = False
            state["current_question"] = check_q
            
            # Save assistant turn
            turn = {
                "turn": state.get("conversation_count", 0) + 1,
                "user_message": None,
                "assistant_message": response_message,
                "stage": "quiz_question",
                "timestamp": datetime.utcnow(),
                "concept_covered": title,
                "message_type": "question",
                "question_number": correct_so_far + 1,
                "progress": f"{correct_so_far}/{required_total}"
            }
            state["conversation_history"].append(turn)
            state["conversation_count"] += 1
            
            return {
                "current_concept_questions_asked": state["current_concept_questions_asked"],
                "expecting_answer": state["expecting_answer"],
                "current_expected_keywords": state["current_expected_keywords"],
                "expecting_button_action": state["expecting_button_action"],
                "current_question": state["current_question"],
                "conversation_history": state["conversation_history"],
                "conversation_count": state["conversation_count"],
                "response": response_message,
                "message_format": "single",
                "is_session_complete": False,
                "current_stage": "quiz_question"
            }
            
        else:
            # Check if this might be a question rather than a button action
            current_concept = state.get("current_question_concept", "")
            current_content = state.get("current_content", "")
            
            # Check relevance of the input
            try:
                relevance = self.rev_agent.check_question_relevance.invoke({
                    "user_input": user_query,
                    "current_concept": current_concept,
                    "content": current_content
                })
                
                if relevance == "RELEVANT":
                    # Handle as relevant question
                    response_message = self.rev_agent.handle_qa_request.invoke({
                        "user_question": user_query,
                        "current_concept": current_concept,
                        "content": current_content,
                        "conversation_history": conv_hist
                    })
                else:
                    # Handle as irrelevant input with polite redirect
                    response_message = self.rev_agent.handle_custom_input.invoke({
                        "user_input": user_query,
                        "current_concept": current_concept,
                        "content": current_content,
                        "conversation_history": conv_hist
                    })
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
            has_used_support = state.get("has_used_learning_support", False)
            
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
                "turn": state.get("conversation_count", 0) + 1 + i,
                "user_message": None,
                "assistant_message": message["assistant_message"],
                "stage": "button_response",
                "timestamp": datetime.utcnow(),
                "concept_covered": title,
                "message_type": message["message_type"],
                "buttons": message.get("buttons", [])
            }
            state["conversation_history"].append(turn)
        
        state["conversation_count"] += len(messages)
        
        return {
            "has_used_learning_support": state["has_used_learning_support"],
            "conversation_history": state["conversation_history"],
            "conversation_count": state["conversation_count"],
            "response": messages,
            "message_format": "multiple_bubbles",
            "is_session_complete": False,
            "current_stage": "button_response"
        }

    def handle_qa_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Handle Q&A requests"""
        conv_hist = self._format_conversation_history(state)
        current_concept = state.get("current_question_concept", "")
        current_chunk_idx = state.get("current_chunk_index", 0)
        concept_chunks = state.get("concept_chunks", [])
        current_content = ""
        
        if current_chunk_idx < len(concept_chunks):
            current_content = concept_chunks[current_chunk_idx].get("content", "")
        
        user_query = state["user_message"]
        # First check relevance
        relevance = self.rev_agent.check_question_relevance.invoke({
            "user_input": user_query,
            "current_concept": current_concept,
            "content": current_content
        })
        
        if relevance == "RELEVANT":
            # Answer the relevant question
            answer = self.rev_agent.handle_qa_request.invoke({
                "user_question": user_query,
                "current_concept": current_concept,
                "content": current_content,
                "conversation_history": conv_hist
            })
            combined = answer.strip()
        else:
            # Handle irrelevant question with polite redirect
            combined = self.rev_agent.handle_custom_input.invoke({
                "user_input": user_query,
                "current_concept": current_concept,
                "content": current_content,
                "conversation_history": conv_hist
            })

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
                "turn": state.get("conversation_count", 0) + 1 + i,
                "user_message": None,
                "assistant_message": message["assistant_message"],
                "stage": "qa",
                "timestamp": datetime.utcnow(),
                "message_type": message["message_type"],
                "buttons": message.get("buttons", [])
            }
            state["conversation_history"].append(turn)
        
        state["conversation_count"] += len(messages)
        state["expecting_button_action"] = True

        return {
            "conversation_history": state["conversation_history"],
            "conversation_count": state["conversation_count"],
            "expecting_button_action": state["expecting_button_action"],
            "response": messages,
            "message_format": "multiple_bubbles",
            "is_session_complete": False,
            "current_stage": "qa"
        }

    def handle_custom_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Handle custom user input that doesn't fit standard categories"""
        conv_hist = self._format_conversation_history(state)
        current_concept = state.get("current_question_concept", "")
        current_chunk_idx = state.get("current_chunk_index", 0)
        concept_chunks = state.get("concept_chunks", [])
        current_content = ""
        
        if current_chunk_idx < len(concept_chunks):
            current_content = concept_chunks[current_chunk_idx].get("content", "")
        
        user_query = state["user_message"]
        # Use revision agent to handle custom input contextually
        response = self.rev_agent.handle_custom_input.invoke({
            "user_input": user_query,
            "current_concept": current_concept,
            "content": current_content,
            "conversation_history": conv_hist
        })
        
        # Create response with appropriate buttons
        buttons_message = "What would you like to do next?"
        
        # Determine which buttons to show based on learning engagement
        has_used_support = state.get("has_used_learning_support", False)
        
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
                "turn": state.get("conversation_count", 0) + 1 + i,
                "user_message": None,
                "assistant_message": message["assistant_message"],
                "stage": "custom_input",
                "timestamp": datetime.utcnow(),
                "message_type": message["message_type"],
                "buttons": message.get("buttons", [])
            }
            state["conversation_history"].append(turn)
        
        state["conversation_count"] += len(messages)
        state["expecting_button_action"] = True

        return {
            "conversation_history": state["conversation_history"],
            "conversation_count": state["conversation_count"],
            "expecting_button_action": state["expecting_button_action"],
            "response": messages,
            "message_format": "multiple_bubbles",
            "is_session_complete": False,
            "current_stage": "custom_input"
        }

    def evaluate_answer_node(self, state: OrchestratorState) -> Dict[str, Any]:
        """Handle answer evaluation for quiz questions"""
        expected_keywords = state.get("current_expected_keywords", [])
        
        # Build full context for evaluation
        conv_hist = self._format_conversation_history(state)
        current_chunk_idx = state.get("current_chunk_index", 0)
        chunks = state.get("concept_chunks", [])
        title = state.get("current_question_concept", "")
        content = ""
        if current_chunk_idx < len(chunks):
            content = chunks[current_chunk_idx].get("content", "")
            
        # Get the current question
        check_question = state.get("current_question", "")
                
        user_query = state["user_message"]
        eval_result = self.rev_agent.evaluate_answer.invoke({
            "user_answer": user_query,
            "expected_keywords": expected_keywords,
            "conversation_history": conv_hist,
            "title": title,
            "content": content,
            "assistant_message": "",
            "check_question": check_question
        })
        
        verdict = eval_result.get("verdict", "WRONG")
        correct_answers = state.get("current_concept_correct_answers", 0)
        required_total = state.get("required_correct_answers", 5)
        
        if verdict == "CORRECT":
            # Check if this is an additional question (after mastery)
            if state.get("concept_mastered", False) and correct_answers >= required_total:
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
                        "turn": state.get("conversation_count", 0) + 1 + i,
                        "user_message": user_query if i == 0 else None,
                        "assistant_message": message["assistant_message"],
                        "stage": "additional_correct",
                        "timestamp": datetime.utcnow(),
                        "correct_answer": True,
                        "is_additional_correct": True,
                        "message_type": message["message_type"],
                        "buttons": message.get("buttons", [])
                    }
                    state["conversation_history"].append(turn)
                
                state["conversation_count"] += len(messages)
                state["expecting_answer"] = False
                state["expecting_button_action"] = True
                
                return {
                    "conversation_history": state["conversation_history"],
                    "conversation_count": state["conversation_count"],
                    "expecting_answer": state["expecting_answer"],
                    "expecting_button_action": state["expecting_button_action"],
                    "response": messages,
                    "message_format": "multiple_bubbles",
                    "is_session_complete": False,
                    "current_stage": "additional_correct"
                }
            
            # Regular progression (not yet mastered)
            # Increment correct answers
            correct_answers += 1
            state["current_concept_correct_answers"] = correct_answers
            
            # Provide positive feedback
            assistant_feedback = f"CORRECT!\nGreat job! Your answer is absolutely right. You covered all the key points:\n\n**Progress: {correct_answers}/{required_total} correct answers**"
            
            # Check if concept is mastered (5 correct answers)
            if correct_answers >= required_total:
                # Concept completed - show recommendation options
                state["concepts_learned"].append(title)
                state["expecting_answer"] = False
                state["expecting_button_action"] = True
                state["concept_mastered"] = True  # Flag to track mastery state
                
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
                        "turn": state.get("conversation_count", 0) + 1 + i,
                        "user_message": user_query if i == 0 else None,
                        "assistant_message": message["assistant_message"],
                        "stage": "concept_mastered",
                        "timestamp": datetime.utcnow(),
                        "correct_answer": True,
                        "concept_mastered": True,
                        "message_type": message["message_type"],
                        "buttons": message.get("buttons", [])
                    }
                    state["conversation_history"].append(turn)
                
                state["conversation_count"] += len(messages)
                
                return {
                    "concepts_learned": state["concepts_learned"],
                    "expecting_answer": state["expecting_answer"],
                    "expecting_button_action": state["expecting_button_action"],
                    "concept_mastered": state["concept_mastered"],
                    "conversation_history": state["conversation_history"],
                    "conversation_count": state["conversation_count"],
                    "response": messages,
                    "message_format": "multiple_bubbles",
                    "is_session_complete": False,
                    "current_stage": "concept_mastered"
                }
            else:
                # Ask next question
                next_question = self.rev_agent.make_check_question.invoke({
                    "title": title,
                    "content": content,
                    "conversation_history": conv_hist
                })
                state["current_concept_questions_asked"].append(next_question)
                
                next_question_message = f"\nLet's try another question:\n\n**Question {correct_answers + 1}:**\n{next_question}"
                combined_message = assistant_feedback + next_question_message
                
                # Update session for next question
                try:
                    expected_keywords = self.rev_agent.extract_expected_keywords.invoke({
                        "title": title,
                        "content": content,
                        "question": next_question
                    })
                except Exception:
                    expected_keywords = [w for w in (title.split()[:3])]
                    
                state["current_expected_keywords"] = expected_keywords
                state["current_question"] = next_question
                
                # Save turns
                feedback_turn = {
                    "turn": state.get("conversation_count", 0) + 1,
                    "user_message": user_query,
                    "assistant_message": combined_message,
                    "stage": "correct_answer_next_question",
                    "timestamp": datetime.utcnow(),
                    "correct_answer": True,
                    "question_number": correct_answers + 1,
                    "progress": f"{correct_answers}/{required_total}"
                }
                state["conversation_history"].append(feedback_turn)
                state["conversation_count"] += 1
                
                return {
                    "current_concept_correct_answers": state["current_concept_correct_answers"],
                    "current_concept_questions_asked": state["current_concept_questions_asked"],
                    "current_expected_keywords": state["current_expected_keywords"],
                    "current_question": state["current_question"],
                    "conversation_history": state["conversation_history"],
                    "conversation_count": state["conversation_count"],
                    "response": combined_message,
                    "message_format": "single",
                    "is_session_complete": False,
                    "current_stage": "next_question"
                }
        else:
            # Wrong/partial answer - provide feedback with retry buttons
            assistant_feedback = self.feedback_agent.feedback_for(verdict, {"correction": eval_result.get("correction", eval_result.get("justification", ""))})
            state["expecting_answer"] = False
            state["expecting_button_action"] = True
            
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
                    "turn": state.get("conversation_count", 0) + 1 + i,
                    "user_message": user_query if i == 0 else None,
                    "assistant_message": message["assistant_message"],
                    "stage": "wrong_answer_feedback",
                    "timestamp": datetime.utcnow(),
                    "correct_answer": False,
                    "message_type": message["message_type"],
                    "buttons": message.get("buttons", [])
                }
                state["conversation_history"].append(turn)
            
            state["conversation_count"] += len(messages)
            
            return {
                "expecting_answer": state["expecting_answer"],
                "expecting_button_action": state["expecting_button_action"],
                "conversation_history": state["conversation_history"],
                "conversation_count": state["conversation_count"],
                "response": messages,
                "message_format": "multiple_bubbles", 
                "is_session_complete": False,
                "current_stage": "wrong_answer_feedback"
            }

    def conclusion_node(self, state: OrchestratorState) -> Dict[str, Any]:
        conv_hist = self._format_conversation_history(state)
        summary = self.conclusion_agent.summary.invoke({
    "correct": len(state.get("concepts_learned", [])),
    "total": len(state.get("concept_chunks", [])),
    "conversation_history": conv_hist
})
        return {
            "response": summary,
            "is_session_complete": True
        }