from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from backend.models.schemas import RevisionRequest, RevisionResponse, TopicResponse
from backend.core.orchestrator_agent import OrchestratorAgent
from backend.core.mongodb_client import MongoDBClient
from datetime import datetime
import uuid
import logging
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependencies - would normally be injected
revision_agent: OrchestratorAgent = None
mongodb_client: MongoDBClient = None

# Thread pool for running sync functions in async context
thread_pool = ThreadPoolExecutor(max_workers=10)

def set_dependencies(ra: OrchestratorAgent, mc: MongoDBClient):
    global revision_agent, mongodb_client
    revision_agent = ra
    mongodb_client = mc

@router.get("/topics", response_model=TopicResponse)
def get_available_topics():
    """
    Get all available topics for revision.
    
    Retrieves a list of all topics available for student revision sessions
    from the MongoDB database. Topics are used to start new revision sessions.
    
    Returns:
        TopicResponse: Object containing list of available topics
        
    Raises:
        HTTPException: 500 status code if database fetch fails
    """
    try:
        topics = mongodb_client.get_available_topics()
        return TopicResponse(topics=topics)
    except Exception as e:
        logger.error(f"Error fetching topics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch topics")

@router.post("/revision/start", response_model=RevisionResponse)
def start_revision_session(request: RevisionRequest):
    """
    Start a new revision session for a specific topic.
    
    Initializes a new learning session with the revision agent for the given topic.
    Generates a unique session ID if not provided and begins the interactive revision
    process tailored to the student's learning needs.
    
    Args:
        request (RevisionRequest): Contains topic, student_id, and optional session_id
        
    Returns:
        RevisionResponse: Initial response with session details, conversation count,
                         completion status, and any relevant sources
                         
    Raises:
        HTTPException: 500 status code if session initialization fails
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Call synchronous method - NO await
        result = revision_agent.start_revision_session(
            topic=request.topic,
            student_id=request.student_id,
            session_id=session_id
        )
        
        return RevisionResponse(
            response=result["response"],
            message_format=result.get("message_format"),
            topic=request.topic,
            session_id=session_id,
            conversation_count=result.get("conversation_count", 0),
            is_session_complete=result["is_session_complete"],
            session_summary=result.get("session_summary"),
            sources=result.get("sources", []),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error starting revision session: {e}")
        raise HTTPException(status_code=500, detail="Failed to start revision session")

@router.post("/revision/continue", response_model=RevisionResponse)
def continue_revision_session(request: RevisionRequest):
    """
    Continue an existing revision session with user input.
    
    Processes user queries within an ongoing revision session, maintaining
    conversation context and tracking learning progress. Updates conversation
    count and determines if the session should be marked as complete.
    
    Args:
        request (RevisionRequest): Contains session_id, user query, and topic
        
    Returns:
        RevisionResponse: AI response with updated session state, progress tracking,
                         completion status, and suggested next actions
                         
    Raises:
        HTTPException: 500 status code if query processing fails
    """
    try:
        # Call synchronous method - NO await
        result = revision_agent.handle_user_input(
            session_id=request.session_id,
            user_query=request.query
        )
        
        return RevisionResponse(
            response=result["response"],
            topic=result.get("topic", request.topic),
            session_id=request.session_id,
            conversation_count=result["conversation_count"],
            is_session_complete=result["is_session_complete"],
            session_summary=result.get("session_summary"),
            next_suggested_action=result.get("next_suggested_action"),
            sources=result.get("sources", []),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error continuing revision session: {e}")
        raise HTTPException(status_code=500, detail="Failed to continue revision session")

# Helper function to run sync code in thread pool
async def run_in_thread(func, *args, **kwargs):
    """Run a synchronous function in a thread pool."""
    loop = asyncio.get_event_loop()
    partial_func = partial(func, *args, **kwargs)
    return await loop.run_in_executor(thread_pool, partial_func)

@router.websocket("/ws/revision/{session_id}")
async def revision_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time revision session communication.
    
    Establishes a persistent WebSocket connection for interactive revision sessions.
    Enables real-time back-and-forth conversation between student and AI tutor,
    with live progress updates and session completion notifications.
    
    Args:
        websocket (WebSocket): WebSocket connection object
        session_id (str): Unique identifier for the revision session
        
    Connection Flow:
        - Accepts WebSocket connection
        - Receives user messages and processes with revision agent
        - Sends AI responses with session metadata
        - Notifies when session is complete with summary
        - Handles disconnection and errors gracefully
        
    Message Types Sent:
        - "message": Regular AI response with conversation data
        - "session_complete": Final summary when revision is finished
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            user_message = await websocket.receive_text()
            logger.info(f"Received message: '{user_message}' for session: {session_id}")  
            
            try:
                # Run synchronous orchestrator in thread pool
                logger.info(f"Processing with revision_agent...")  
                result = await run_in_thread(
                    revision_agent.handle_user_input,
                    session_id,
                    user_message
                )
                
                logger.info(f"Got result: {type(result)} - {result.get('response', 'No response')[:50]}...")
                logger.info(f"WebSocket result message_format: {result.get('message_format')}")  
                
                # Prepare response data
                response_data = {
                    "type": "message",
                    "content": result["response"],
                    "message_format": result.get("message_format", "single"),
                    "conversation_count": result.get("conversation_count", 0),    
                    "is_session_complete": result.get("is_session_complete", False), 
                    "current_stage": result.get("current_stage", "revision"),    
                    "sources": result.get("sources", [])                          
                }
                
                # Send response
                await websocket.send_text(json.dumps(response_data))
                
                # Send completion message if session is complete
                if result.get("is_session_complete", False):
                    complete_data = {
                        "type": "session_complete",
                        "summary": result.get("session_summary", "Session completed successfully!")
                    }
                    await websocket.send_text(json.dumps(complete_data))
                    break  # End the WebSocket connection
                    
            except Exception as processing_error:
                logger.error(f"Error processing message: {processing_error}")
                
                # Send error message to client
                error_data = {
                    "type": "error",
                    "content": "I'm having trouble processing your message. Please try again.",
                    "error_details": str(processing_error) if logger.isEnabledFor(logging.DEBUG) else None
                }
                
                try:
                    await websocket.send_text(json.dumps(error_data))
                except Exception as send_error:
                    logger.error(f"Failed to send error message: {send_error}")
                    break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass  # Connection might already be closed

# Health check endpoint for the API
@router.get("/health")
def health_check():
    """Health check endpoint to verify API is running."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "revision_agent_status": "connected" if revision_agent else "not_connected",
        "mongodb_status": "connected" if mongodb_client else "not_connected"
    }

# Endpoint to get session information
@router.get("/revision/session/{session_id}")
def get_session_info(session_id: str):
    """Get information about a specific revision session."""
    try:
        session_data = mongodb_client.get_revision_session(session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Remove sensitive or large data before returning
        safe_session_data = {
            "session_id": session_data.get("session_id"),
            "topic": session_data.get("topic"),
            "student_id": session_data.get("student_id"),
            "started_at": session_data.get("started_at"),
            "conversation_count": session_data.get("conversation_count", 0),
            "is_complete": session_data.get("is_complete", False),
            "current_stage": session_data.get("current_stage", "unknown"),
            "concepts_learned": session_data.get("concepts_learned", [])
        }
        
        return safe_session_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching session info: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch session information")

# Endpoint to end a session
@router.post("/revision/end/{session_id}")
def end_revision_session(session_id: str):
    """Manually end a revision session."""
    try:
        session_data = mongodb_client.get_revision_session(session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session_data.get("is_complete", False):
            return {"message": "Session already completed", "session_id": session_id}
        
        # Mark session as complete
        update_data = {
            "is_complete": True,
            "completed_at": datetime.now(),
            "session_summary": "Session ended manually by user."
        }
        
        success = mongodb_client.update_session_progress(session_id, update_data)
        
        if success:
            return {"message": "Session ended successfully", "session_id": session_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to end session")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending session: {e}")
        raise HTTPException(status_code=500, detail="Failed to end session")

# Get statistics for all sessions
@router.get("/revision/stats")
def get_revision_stats():
    """Get overall revision statistics."""
    try:
        # This would require additional methods in mongodb_client
        # For now, return basic stats
        return {
            "message": "Statistics endpoint - implementation depends on mongodb_client methods",
            "timestamp": datetime.now().isoformat(),
            "status": "available"
        }
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch statistics")