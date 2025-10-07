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
    next_action: Optional[str]

class OrchestratorAgent:
    def __init__(self, mongodb: Optional[MongoDBClient] = None):
        self.rev_agent = RevisionAgent()
        self.quiz_agent = QuizAgent()
        self.feedback_agent = FeedbackAgent()
        self.qa_agent = QAAgent()
        self.conclusion_agent = ConclusionAgent()
        self.mongo = mongodb or MongoDBClient()

        self.graph = StateGraph(OrchestratorState)
        self._build_graph()
        self.app = self.graph.compile()

    def _build_graph(self):
        g = self.graph
        g.add_node("handle_input", self.handle_input_node)
        g.add_node("detect_intent", self.detect_intent_node)
        g.add_node("handle_ack", self.handle_ack_node)
        g.add_node("handle_qa", self.handle_qa_node)
        g.add_node("handle_custom", self.handle_custom_node)
        g.add_node("handle_button", self.handle_button_node)
        g.add_node("evaluate_answer", self.evaluate_answer_node)
        g.add_node("present_concept", self.present_concept_node)
        g.add_node("conclusion", self.conclusion_node)

        g.set_conditional_entry_point(
            self.entry_route,
            path_map={
                "present_concept": "present_concept",
                "handle_input": "handle_input",
            },
        )

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

        g.add_conditional_edges(
            "detect_intent",
            self.route_after_detect,
            {
                "handle_ack": "handle_ack",
                "handle_qa": "handle_qa",
                "handle_custom": "handle_custom",
            },
        )

        g.add_conditional_edges(
            "handle_button",
            self.route_after_button,
            {
                "present_concept": "present_concept",
                "end": END,
            },
        )

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
        next_action = state.get("next_action")
        logger.info(f"Routing after button action: next_action={next_action}")
        if next_action == "present_concept":
            return "present_concept"
        return "end"

    def _format_conversation_history(self, state: Dict[str, Any], limit: int = 10) -> str:
        if not state:
            return ""
        conv = state.get("conversation_history", [])[-limit:]
        conv = list(reversed(conv))
        lines = []
        for i, turn in enumerate(conv):
            user = turn.get("user_message", "")
            assistant = turn.get("assistant_message", "")
            ts = turn.get("timestamp", "")
            lines.append(f"[{i}] user: {user} | assistant: {assistant}")
        return "\n".join(lines)

    def start_revision_session(self, topic: str, student_id: str, session_id: str) -> Dict[str, Any]:
        session_doc = self.mongo.get_revision_session(session_id) or {}
        if session_doc and session_doc.get("is_complete", False):
            logger.warning(f"Session {session_id} already exists and is complete")
            return {"response": "Session already completed. Start a new session.", "is_session_complete": True, "conversation_count": 0}

        if not session_doc:
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
                "current_concept_correct_answers": 0,
                "required_correct_answers": 2,
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

        topic_title = topic.split(": ")[-1] if ": " in topic else topic
        subtopics = self.mongo.get_topic_subtopics(topic_title)
        if not subtopics:
            subtopic_chunks = self.mongo.get_topic_content(topic)
            subtopics = [{"subtopic_number": c["id"], "subtopic_title": c.get("subtopic_title", ""), "content": c["text"]} for c in subtopic_chunks]

        session_doc["concept_chunks"] = subtopics
        session_doc["current_chunk_index"] = 0

        state = OrchestratorState(**session_doc)
        output_state = self.app.invoke(state)

        self.mongo.save_revision_session(output_state)

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
            return {"response": "Session not found. Start a new revision session.", "is_session_complete": True, "conversation_count": 0}

        state = OrchestratorState(**session_doc)
        state["user_message"] = user_query

        output_state = self.app.invoke(state)

        self.mongo.save_revision_session(output_state)

        result = {
            "response": output_state["response"],
            "message_format": output_state.get("message_format"),
            "is_session_complete": output_state["is_session_complete"],
            "conversation_count": output_state["conversation_count"],
            "current_stage": output_state["current_stage"]
        }
        return result

    def present_concept_node(self, state: OrchestratorState) -> Dict[str, Any]:
        if state.get("current_stage") == "explain" and state.get("response"):
            logger.info("Skipping redundant concept presentation")
            return state

        idx = state.get("current_chunk_index", 0)
        chunks = state.get("concept_chunks", [])
        if idx >= len(chunks):
            logger.info(f"No more concepts to present. Total chunks: {len(chunks)}. Ending session.")
            conv_hist = self._format_conversation_history(state)
            try:
                summary = self.conclusion_agent.summary.invoke({
                    "correct": len(state.get("concepts_learned", [])),
                    "total": len(chunks),
                    "conversation_history": conv_hist
                })
            except Exception as e:
                logger.error(f"Failed to generate summary: {e}")
                summary = f"Session completed. You mastered {len(state.get('concepts_learned', []))} out of {len(chunks)} concepts."
            
            updates = {
                "is_complete": True,
                "response": summary,
                "is_session_complete": True,
                "conversation_count": state.get("conversation_count", 0),
                "current_stage": "completed"
            }
            return updates
        
        current = chunks[idx]
        title = current.get("subtopic_title") or f"Concept {current.get('subtopic_number')}"
        content = current.get("content", "")
        
        state["current_concept_correct_answers"] = 0
        state["current_concept_questions_asked"] = []
        state["has_used_learning_support"] = False
        
        conv_hist = self._format_conversation_history(state)
        structured_content = self.rev_agent.generate_structured_explanation.invoke({
            "title": title, 
            "content": content, 
            "conversation_history": conv_hist
        })
        
        if not isinstance(structured_content, list) or not all(isinstance(msg, dict) for msg in structured_content):
            logger.warning(f"Invalid structured content for {title}: {structured_content}")
            structured_content = [{"assistant_message": content, "message_type": "response"}]
        
        messages = structured_content.copy()
        
        buttons_message = "What would you like to do next?"
        messages.append({
            "assistant_message": buttons_message, 
            "message_type": "buttons",
            "buttons": [
                {"text": "I need more examples", "action": "more_examples"},
                {"text": "Can you re-explain?", "action": "re_explain"}
            ]
        })
        
        existing_response = state.get("response", [])
        if isinstance(existing_response, list):
            transition_messages = [msg for msg in existing_response if msg.get("message_type") == "transition"]
            if transition_messages:
                messages = transition_messages + messages
        
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

        state["conversation_count"] = state.get("conversation_count", 0) + len(messages)
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
        logger.info(f"Entering handle_button_node with user_message: {state.get('user_message')}")
        
        current_chunk_idx = state.get("current_chunk_index", 0)
        chunks = state.get("concept_chunks", [])
        
        if current_chunk_idx >= len(chunks):
            logger.info("No more concepts to explore, marking session complete")
            conv_hist = self._format_conversation_history(state)
            try:
                summary = self.conclusion_agent.summary.invoke({
                    "correct": len(state.get("concepts_learned", [])),
                    "total": len(chunks),
                    "conversation_history": conv_hist
                })
            except Exception as e:
                logger.error(f"Failed to generate summary: {e}")
                summary = f"Session completed. You mastered {len(state.get('concepts_learned', []))} out of {len(chunks)} concepts."
            
            return {
                "response": summary,
                "conversation_count": state["conversation_count"],
                "is_session_complete": True,
                "current_stage": "completed"
            }
        
        current = chunks[current_chunk_idx]
        title = current.get("subtopic_title") or f"Concept {current.get('subtopic_number')}"
        content = current.get("content", "")
        user_query = state.get("user_message", "").lower().strip()
        
        conv_hist = self._format_conversation_history(state)
        response_message = ""
        next_action = "continue"
        
        if state.get("concept_mastered", False):
            logger.info(f"Concept mastered, handling mastery button: {user_query}")
            if user_query in ["more_questions", "more questions"]:
                state["concept_mastered"] = False
                state["expecting_button_action"] = False
                state["expecting_answer"] = True
                
                next_question = self.rev_agent.make_check_question.invoke({
                    "title": title,
                    "content": content,
                    "conversation_history": conv_hist
                })
                if not isinstance(next_question, str) or not next_question.strip():
                    logger.warning(f"Invalid question format for {title}")
                    next_question = f"What is the main idea of {title}?"
                
                state["current_concept_questions_asked"].append(next_question)
                
                correct_answers = state.get("current_concept_correct_answers", 0)
                response_message = f"Great! Let's continue with more questions to deepen your understanding.\n\n**Additional Question {correct_answers + 1}:**\n{next_question}"
                
                try:
                    expected_keywords = self.rev_agent.extract_expected_keywords.invoke({
                        "title": title,
                        "content": content,
                        "question": next_question
                    })
                except Exception as e:
                    logger.warning(f"Failed to extract keywords: {e}")
                    expected_keywords = [w for w in title.split()[:3]]
                
                state["current_expected_keywords"] = expected_keywords
                state["current_question"] = next_question
                
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
                
                logger.info(f"Generated additional question for {title}: {next_question[:100]}...")
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
            
            elif user_query in ["next_concept", "next concept"]:
                state["concept_mastered"] = False
                state["current_chunk_index"] += 1
                state["expecting_answer"] = False
                state["expecting_button_action"] = False
                state["current_question_concept"] = None
                state["current_content"] = ""
                state["current_concept_correct_answers"] = 0
                state["current_concept_questions_asked"] = []
                
                transition_message = "Perfect! Moving to the next concept..."
                state["response"] = [{"assistant_message": transition_message, "message_type": "transition"}]
                state["message_format"] = "multiple_bubbles"
                
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
                
                state["next_action"] = "present_concept"
                
                logger.info(f"Transitioning to next concept at index {state['current_chunk_index']}, reset quiz progress")
                return {
                    "concept_mastered": state["concept_mastered"],
                    "current_chunk_index": state["current_chunk_index"],
                    "expecting_answer": state["expecting_answer"],
                    "expecting_button_action": state["expecting_button_action"],
                    "current_question_concept": state["current_question_concept"],
                    "current_content": state["current_content"],
                    "current_concept_correct_answers": state["current_concept_correct_answers"],
                    "current_concept_questions_asked": state["current_concept_questions_asked"],
                    "response": state["response"],
                    "message_format": state["message_format"],
                    "conversation_history": state["conversation_history"],
                    "conversation_count": state["conversation_count"],
                    "next_action": state["next_action"],
                    "is_session_complete": False,
                    "current_stage": "concept_transition"
                }
        
        if user_query in ["more_examples", "more examples"]:
            response_message = self.rev_agent.generate_examples.invoke({
                "title": title,
                "content": content,
                "conversation_history": conv_hist
            })
            if not isinstance(response_message, str) or not response_message.strip():
                logger.warning(f"Invalid examples format for {title}")
                response_message = f"Fallback example: {content[:100]}"
            next_action = "continue"
            state["has_used_learning_support"] = True
            logger.info(f"Generated examples for {title}: {response_message[:100]}...")
            
        elif user_query in ["re_explain", "re-explain"]:
            steps = self.rev_agent.generate_explanation_steps.invoke({
                "title": title,
                "content": content,
                "conversation_history": conv_hist,
                "steps": 4
            })
            if not isinstance(steps, list) or len(steps) != 4:
                logger.warning(f"Invalid steps format for {title}: {steps}")
                steps = [f"Fallback step {i+1}: {content[:50]}" for i in range(4)]
            response_message = "Let me explain this concept again in a different way:\n\n" + "\n".join(steps)
            next_action = "continue"
            state["has_used_learning_support"] = True
            logger.info(f"Generated step-by-step explanation for {title}: {response_message[:100]}...")
            
        elif user_query in ["check_understanding", "check my understanding"]:
            check_q = self.rev_agent.make_check_question.invoke({
                "title": title,
                "content": content,
                "conversation_history": conv_hist
            })
            if not isinstance(check_q, str) or not check_q.strip():
                logger.warning(f"Invalid question format for {title}")
                check_q = f"What is the main idea of {title}?"
            
            state["current_concept_questions_asked"].append(check_q)
            correct_so_far = state.get("current_concept_correct_answers", 0)
            required_total = state.get("required_correct_answers", 5)
            
            response_message = f"Great! Let's test your understanding. You need to answer {required_total} questions correctly to master this concept.\n\n**Progress: {correct_so_far}/{required_total} correct answers**\n\n**Question {correct_so_far + 1}:**\n{check_q}"
            
            try:
                expected_keywords = self.rev_agent.extract_expected_keywords.invoke({
                    "title": title,
                    "content": content,
                    "question": check_q
                })
            except Exception as e:
                logger.warning(f"Failed to extract keywords: {e}")
                expected_keywords = [w for w in title.split()[:3]]
            
            state["expecting_answer"] = True
            state["current_expected_keywords"] = expected_keywords
            state["expecting_button_action"] = False
            state["current_question"] = check_q
            
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
            
            logger.info(f"Generated quiz question for {title}: {check_q[:100]}...")
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
            logger.info(f"Handling unrecognized input: {user_query}")
            relevance = self.rev_agent.check_question_relevance.invoke({
                "user_input": user_query,
                "current_concept": title,
                "content": content
            })
            
            if relevance == "RELEVANT":
                response_message = self.rev_agent.handle_qa_request.invoke({
                    "user_question": user_query,
                    "current_concept": title,
                    "content": content,
                    "conversation_history": conv_hist
                })
                logger.info(f"Generated QA response for {title}: {response_message[:100]}...")
            else:
                # STRICT REDIRECT
                response_message = f"⚠️ **That's off-topic.**\n\nRight now we're focusing on **{title}**. Please ask questions about this concept or use the buttons below to continue."
                logger.info(f"IRRELEVANT question blocked for {title}: {user_query[:100]}")
            
            if not isinstance(response_message, str) or not response_message.strip():
                logger.warning(f"Invalid response format for {title}")
                response_message = f"⚠️ **Let's stay focused on {title}.**"
            next_action = "continue"
        
        messages = [{"assistant_message": response_message, "message_type": "response"}]
        
        if next_action == "continue":
            buttons_message = "What would you like to do next?"
            has_used_support = state.get("has_used_learning_support", False)
            buttons = [
                {"text": "I need more examples", "action": "more_examples"},
                {"text": "Can you re-explain?", "action": "re_explain"}
            ]
            if has_used_support:
                buttons.append({"text": "Let me check my understanding with some Q&A", "action": "check_understanding"})
            
            messages.append({
                "assistant_message": buttons_message,
                "message_type": "buttons",
                "buttons": buttons
            })
        
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
        
        logger.info(f"Returning response with {len(messages)} messages, stage: button_response")
        return {
            "has_used_learning_support": state["has_used_learning_support"],
            "conversation_history": state["conversation_history"],
            "conversation_count": state["conversation_count"],
            "response": messages,
            "message_format": "multiple_bubbles",
            "is_session_complete": False,
            "current_stage": "button_response",
            "next_action": next_action
        }

    def handle_qa_node(self, state: OrchestratorState) -> Dict[str, Any]:
        conv_hist = self._format_conversation_history(state)
        current_concept = state.get("current_question_concept", "")
        current_chunk_idx = state.get("current_chunk_index", 0)
        concept_chunks = state.get("concept_chunks", [])
        current_content = ""
        
        if current_chunk_idx < len(concept_chunks):
            current_content = concept_chunks[current_chunk_idx].get("content", "")
        
        user_query = state["user_message"]
        user_query_lower = user_query.lower()
        
        # STRICT: Block questions about people/celebrities
        person_keywords = ["who is", "who's", "what is his", "what is her", "celebrity", "actor", "actress", "star", "famous person"]
        if any(keyword in user_query_lower for keyword in person_keywords):
            combined = f"⚠️ **That question is off-topic.**\n\nWe're currently learning about **{current_concept}**. Questions about people, celebrities, or trivia are not part of this lesson.\n\nPlease ask questions related to **{current_concept}**, or use the buttons below to continue learning."
            logger.info(f"BLOCKED person/celebrity question for {current_concept}: {user_query[:100]}")
        else:
            # Continue with relevance check
            relevance = self.rev_agent.check_question_relevance.invoke({
                "user_input": user_query,
                "current_concept": current_concept,
                "content": current_content
            })
            
            if relevance == "RELEVANT":
                answer = self.rev_agent.handle_qa_request.invoke({
                    "user_question": user_query,
                    "current_concept": current_concept,
                    "content": current_content,
                    "conversation_history": conv_hist
                })
                logger.info(f"Generated QA response for {current_concept}: {answer[:100]}...")
                combined = answer.strip()
            else:
                combined = f"⚠️ **That question is off-topic.**\n\nWe're currently learning about **{current_concept}**. Let's stay focused on this concept.\n\nPlease ask questions related to **{current_concept}**, or use the buttons below to continue learning."
                logger.info(f"IRRELEVANT question blocked for {current_concept}: {user_query[:100]}")
        
    # Rest of the function remains the same...
        
        if not isinstance(combined, str) or not combined.strip():
            logger.warning(f"Invalid QA/custom response format for {current_concept}")
            combined = f"⚠️ **Let's stay on topic.**\n\nWe're learning about **{current_concept}**. Please ask questions related to this concept."

        buttons_message = "What would you like to do next?"
        has_used_support = state.get("has_used_learning_support", False)
        buttons = [
            {"text": "I need more examples", "action": "more_examples"},
            {"text": "Can you re-explain?", "action": "re_explain"}
        ]
        if has_used_support:
            buttons.append({"text": "Let me check my understanding with some Q&A", "action": "check_understanding"})

        messages = [
            {"assistant_message": combined, "message_type": "qa_response"},
            {
                "assistant_message": buttons_message,
                "message_type": "buttons",
                "buttons": buttons
            }
        ]

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
        current_concept = state.get("current_question_concept", "")
        
        # STRICT: Direct redirect without checking relevance
        response = f"⚠️ **Let's stay on topic!**\n\nWe're learning about **{current_concept}** right now. Please focus on this concept."
        
        logger.info(f"Custom/irrelevant input redirected for {current_concept}: {state['user_message'][:100]}")

        buttons_message = "What would you like to do next?"
        has_used_support = state.get("has_used_learning_support", False)
        buttons = [
            {"text": "I need more examples", "action": "more_examples"},
            {"text": "Can you re-explain?", "action": "re_explain"}
        ]
        if has_used_support:
            buttons.append({"text": "Let me check my understanding with some Q&A", "action": "check_understanding"})

        messages = [
            {"assistant_message": response, "message_type": "custom_response"},
            {
                "assistant_message": buttons_message,
                "message_type": "buttons",
                "buttons": buttons
            }
        ]

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
        expected_keywords = state.get("current_expected_keywords", [])
        
        conv_hist = self._format_conversation_history(state)
        current_chunk_idx = state.get("current_chunk_index", 0)
        chunks = state.get("concept_chunks", [])
        title = state.get("current_question_concept", "")
        content = ""
        if current_chunk_idx < len(chunks):
            content = chunks[current_chunk_idx].get("content", "")
            
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
            if state.get("concept_mastered", False) and correct_answers >= required_total:
                assistant_feedback = f"✅ **CORRECT!**\n\n🎉 Excellent! You continue to demonstrate strong understanding of this concept.\n\n**What you got right:**\n{eval_result.get('justification', 'Great explanation!')}"
                
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
            
            correct_answers += 1
            state["current_concept_correct_answers"] = correct_answers
            
            assistant_feedback = f"✅ **CORRECT!**\n\n🎉 Great job! Your answer is absolutely right.\n\n**What you understood well:**\n{eval_result.get('justification', 'You covered all the key points!')}\n\n**Progress: {correct_answers}/{required_total} correct answers**"
            
            if correct_answers >= required_total:
                state["concepts_learned"].append(title)
                state["expecting_answer"] = False
                state["expecting_button_action"] = True
                state["concept_mastered"] = True
                
                completion_message = f"\n\n🏆 **Concept Mastered!**\nYou've successfully answered {required_total} questions correctly. You've shown excellent understanding of **{title}**!"
                assistant_feedback += completion_message
                
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
                next_question = self.rev_agent.make_check_question.invoke({
                    "title": title,
                    "content": content,
                    "conversation_history": conv_hist
                })
                if not isinstance(next_question, str) or not next_question.strip():
                    logger.warning(f"Invalid question format for {title}")
                    next_question = f"What is the main idea of {title}?"
                
                state["current_concept_questions_asked"].append(next_question)
                
                next_question_message = f"\nLet's try another question:\n\n**Question {correct_answers + 1}:**\n{next_question}"
                combined_message = assistant_feedback + next_question_message
                
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
        elif verdict == "PARTIAL":
            assistant_feedback = f"🟡 **PARTIALLY CORRECT**\n\n👍 Good effort! You're on the right track, but there are some missing elements.\n\n**What you got right:**\n{eval_result.get('justification', 'You have some understanding.')}\n\n**What to add:**\n{eval_result.get('correction', 'Try to include more key details.')}"
            
            state["expecting_answer"] = False
            state["expecting_button_action"] = True
            
            feedback_message = assistant_feedback
            buttons_message = "Let's strengthen your understanding. What would you like to do?"

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
            
            for i, message in enumerate(messages):
                turn = {
                    "turn": state.get("conversation_count", 0) + 1 + i,
                    "user_message": user_query if i == 0 else None,
                    "assistant_message": message["assistant_message"],
                    "stage": "partial_answer_feedback",
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
                "current_stage": "partial_answer_feedback"
            }
        else:
            assistant_feedback = f"❌ **INCORRECT**\n\n💭 Not quite right, but that's okay - learning involves making mistakes!\n\n**What needs correction:**\n{eval_result.get('correction', eval_result.get('justification', 'Let\'s review this concept together.'))}"
            
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
        try:
            summary = self.conclusion_agent.summary.invoke({
                "correct": len(state.get("concepts_learned", [])),
                "total": len(state.get("concept_chunks", [])),
                "conversation_history": conv_hist
            })
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            summary = f"Session completed. You mastered {len(state.get('concepts_learned', []))} out of {len(state.get('concept_chunks', []))} concepts."
        
        return {
            "response": summary,
            "is_session_complete": True,
            "current_stage": "completed"
        }