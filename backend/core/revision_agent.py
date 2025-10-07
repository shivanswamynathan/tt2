from typing import List, Dict, Any, Optional
from .llm import GeminiLLMWrapper
from backend.prompts import revision_prompts
from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)
llm_wrapper = GeminiLLMWrapper()

# Standalone tool functions (outside the class)
@tool
def generate_structured_explanation(title: str, content: str, conversation_history: str = "") -> List[Dict[str, Any]]:
    """Generate structured explanation with multiple detailed bubbles for a concept."""
    
    # Enhanced prompt - uses ||| separator instead of headers
    enhanced_prompt = f"""{revision_prompts.STRUCTURED_EXPLANATION_TEMPLATE.format(
        title=title, content=content, conversation_history=conversation_history
    )}

CRITICAL FORMATTING REQUIREMENTS:
1. Separate the 3 messages with "|||" (three pipes)
2. MESSAGE 1: ONLY concept name - MAXIMUM 20 CHARACTERS
3. MESSAGE 2: Full detailed explanation (100+ words)
4. MESSAGE 3: 2-3 concrete examples with details
5. Do NOT include labels like "MESSAGE 1:", "BUBBLE_1:", etc. in the output
6. Use proper markdown formatting with ** for bold text
7. Include emojis as shown in the template

Example output format:
💡 **Gravity**

|||

📖 **What is Gravity?**

Gravity is the fundamental force...
[Full explanation continues]

|||

🌟 **Examples of Gravity**

🍎 **Example 1:** Falling objects...
🌍 **Example 2:** Planetary orbits...
"""
    
    logger.info(f"Generating structured explanation for: {title}")
    resp = llm_wrapper.generate_response([{"role":"user","content": enhanced_prompt}])
    
    logger.info(f"LLM Response length: {len(resp)} characters")
    logger.info(f"Response preview: {resp[:200]}...")
    
    bubbles = _parse_detailed_bubbles(resp, title, content)
    
    logger.info(f"Generated {len(bubbles)} bubbles")
    for i, bubble in enumerate(bubbles):
        logger.info(f"Bubble {i+1} ({bubble['section']}): {len(bubble['assistant_message'])} chars")
    
    return bubbles

def _parse_detailed_bubbles(response: str, title: str, content: str) -> List[Dict[str, Any]]:
    """Parse LLM response into 3 structured bubbles using ||| separator"""
    bubbles = []
    
    # Split by the separator |||
    messages = response.split('|||')
    
    # Clean up each message
    messages = [msg.strip() for msg in messages if msg.strip()]
    
    logger.info(f"Parsed {len(messages)} messages from LLM response")
    
    # We expect exactly 3 messages
    if len(messages) == 3:
        # Message 1: Concept Name
        bubbles.append({
            "assistant_message": messages[0],
            "message_type": "concept_section",
            "section": "Concept Name"
        })
        
        # Message 2: Explanation
        bubbles.append({
            "assistant_message": messages[1],
            "message_type": "concept_section",
            "section": "Explanation"
        })
        
        # Message 3: Examples
        bubbles.append({
            "assistant_message": messages[2],
            "message_type": "concept_section",
            "section": "Examples"
        })
        
        logger.info(f"Successfully created 3 bubbles from parsed messages")
        for i, bubble in enumerate(bubbles):
            logger.info(f"Bubble {i+1}: {len(bubble['assistant_message'])} chars")
    else:
        # Fallback if AI didn't use the separator correctly
        logger.warning(f"Expected 3 messages but got {len(messages)}. Using fallback.")
        bubbles = _create_detailed_fallback(content, title)
    
    return bubbles


def _create_detailed_fallback(content: str, title: str) -> List[Dict[str, Any]]:
    """Create detailed fallback structure using actual MongoDB content - PRESERVES ALL CONTENT"""
    bubbles = []
    
    content = content.strip()
    
    # Handle edge case: no content
    if not content or len(content) < 50:
        logger.warning(f"Content too short ({len(content)} chars), using placeholders")
        return _create_placeholder_bubbles(title)
    
    logger.info(f"Creating fallback from content ({len(content)} chars)")
    
    # STRATEGY: Split by paragraphs (double newline)
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    # If not enough paragraphs, split by single newline
    if len(paragraphs) < 2:
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
    
    # If still not enough, use the whole content
    if len(paragraphs) < 2:
        import re
        sentences = re.split(r'[.!?]\s+', content)
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        if len(sentences) >= 2:
            # Use first few sentences for explanation, rest for examples
            explanation_text = ' '.join(sentences[:max(1, len(sentences)//2)])
            examples_text = ' '.join(sentences[max(1, len(sentences)//2):])
        else:
            explanation_text = content
            examples_text = content
    else:
        # Use first paragraph(s) for explanation, last for examples
        mid_point = max(1, len(paragraphs)//2)
        explanation_text = '\n\n'.join(paragraphs[:mid_point])
        examples_text = '\n\n'.join(paragraphs[mid_point:])
    
    # BUBBLE 1: Concept Name ONLY (max 20 characters)
    # Extract just the key term from title
    concept_name = title.replace("**", "").strip()
    # Truncate to 20 characters if needed
    if len(concept_name) > 20:
        words = concept_name.split()
        concept_name = words[0] if words else concept_name[:20]
        # Make sure it's still under 20 chars
        if len(concept_name) > 20:
            concept_name = concept_name[:20]
    
    bubbles.append({
        "assistant_message": f"💡 **{concept_name}**",
        "message_type": "concept_section",
        "section": "Concept Name"
    })
    
    # BUBBLE 2: Explanation
    explanation_content = f"📖 **What is {title}?**\n\n{explanation_text}"
    
    bubbles.append({
        "assistant_message": explanation_content,
        "message_type": "concept_section",
        "section": "Explanation"
    })
    
    # BUBBLE 3: Examples
    examples_content = f"🌟 **Examples of {title}**\n\n{examples_text}"
    
    bubbles.append({
        "assistant_message": examples_content,
        "message_type": "concept_section",
        "section": "Examples"
    })
    
    # Log final bubble sizes
    for i, bubble in enumerate(bubbles):
        logger.info(f"Fallback Bubble {i+1}: {len(bubble['assistant_message'])} chars")
    
    return bubbles


def _create_placeholder_bubbles(title: str) -> List[Dict[str, Any]]:
    """Create placeholder bubbles when content is insufficient"""
    # Extract concept name (max 20 characters)
    concept_name = title.replace("**", "").strip()
    if len(concept_name) > 20:
        words = concept_name.split()
        concept_name = words[0] if words else concept_name[:20]
        # Ensure it's under 20 chars
        if len(concept_name) > 20:
            concept_name = concept_name[:20]
    
    return [
        {
            "assistant_message": f"💡 **{concept_name}**",
            "message_type": "concept_section",
            "section": "Concept Name"
        },
        {
            "assistant_message": f"📖 **What is {title}?**\n\n**{title}** is an important concept in this topic that helps us understand fundamental principles. It involves understanding the key mechanisms and processes that make this concept work in practice.",
            "message_type": "concept_section",
            "section": "Explanation"
        },
        {
            "assistant_message": f"🌟 **Examples of {title}**\n\n🏠 **Example 1:** Everyday applications of {title.lower()}\n\n🔬 **Example 2:** Scientific demonstrations of {title.lower()}",
            "message_type": "concept_section",
            "section": "Examples"
        }
    ]


@tool
def generate_explanation_steps(title: str, content: str, conversation_history: str = "", steps: int = 4) -> List[str]:
    """Generate step-by-step explanation"""
    prompt = revision_prompts.EXPLANATION_TEMPLATE.format(
        title=title, steps=steps, content=content, conversation_history=conversation_history
    )
    resp = llm_wrapper.generate_response([{"role":"user","content": prompt}])
    
    logger.info(f"Re-explain response length: {len(resp)} chars")
    
    # Parse steps from response
    step_lines = []
    current_step = []
    
    for line in resp.split('\n'):
        line_stripped = line.strip()
        
        # Detect step markers
        if any(marker in line_stripped.lower() for marker in ['🔸 **step', 'step ', '**step']):
            if current_step:
                step_lines.append('\n'.join(current_step).strip())
            current_step = [line]
        elif current_step:
            current_step.append(line)
    
    # Add last step
    if current_step:
        step_lines.append('\n'.join(current_step).strip())
    
    # Fallback if parsing failed
    if not step_lines:
        step_lines = [l.strip() for l in resp.splitlines() if l.strip()]
    
    # Ensure we have exactly 'steps' items
    while len(step_lines) < steps:
        step_lines.append(f"**Step {len(step_lines)+1}:** Continue exploring this concept...")
    
    return step_lines[:steps]


@tool  
def generate_examples(title: str, content: str, conversation_history: str = "") -> str:
    """Generate practical examples for the given concept."""
    
    # Calculate content length and determine number of examples
    content_length = len(content)
    num_examples = 3 if content_length <= 100 else 2
    
    # Prepare example_3 placeholder (only if num_examples == 3)
    example_3 = "🌟 **Example 3:** [Creative or unexpected example]\n→ This showcases **{title}** in [unique context or application]" if num_examples == 3 else ""
    
    prompt = revision_prompts.EXAMPLES_TEMPLATE.format(
        title=title, 
        content=content, 
        conversation_history=conversation_history,
        num_examples=num_examples,
        content_length=content_length,
        example_3=example_3
    )
    resp = llm_wrapper.generate_response([{"role":"user","content": prompt}])
    return resp.strip()


@tool
def make_check_question(title: str, content: str, conversation_history: str = "") -> str:
    """Generate a check question to test understanding."""
    prompt = revision_prompts.CHECK_QUESTION_TEMPLATE.format(
        title=title, content=content, conversation_history=conversation_history
    )
    resp = llm_wrapper.generate_response([{"role":"user","content": prompt}])
    return resp.strip()


@tool
def extract_expected_keywords(title: str, content: str, question: str) -> List[str]:
    """Extract expected keywords for answer evaluation."""
    prompt = revision_prompts.KEYWORDS_EXTRACTION_TEMPLATE.format(
        title=title, content=content, question=question
    )
    resp = llm_wrapper.generate_response([{"role":"user","content": prompt}])
    text = resp.strip()
    
    try:
        import json
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip().lower() for x in data if str(x).strip()]
    except Exception:
        pass
    
    return [w.lower() for w in title.split()[:3]]


@tool
def evaluate_answer(user_answer: str, expected_keywords: List[str], conversation_history: str = "", 
                   title: str = "", content: str = "", assistant_message: str = "", 
                   check_question: str = "") -> Dict[str, Any]:
    """Evaluate user answer using full context."""
    prompt = revision_prompts.EVAL_WITH_CONTEXT_TEMPLATE.format(
        title=title, content=content, assistant_message=assistant_message,
        check_question=check_question, user_answer=user_answer
    )
    resp = llm_wrapper.generate_response([{"role":"user","content": prompt}])
    
    verdict = "WRONG"
    justification = ""
    correction = ""
    
    for line in resp.splitlines():
        line_low = line.strip()
        if line_low.upper().startswith("VERDICT:"):
            verdict = line_low.split(":",1)[1].strip().upper()
        elif line_low.upper().startswith("JUSTIFICATION:"):
            justification = line_low.split(":",1)[1].strip()
        elif line_low.upper().startswith("CORRECTION:"):
            correction = line_low.split(":",1)[1].strip()
    
    return {
        "verdict": verdict,
        "justification": justification,
        "correction": correction or justification
    }


@tool
def handle_qa_request(user_question: str, current_concept: str, content: str, 
                     conversation_history: str = "") -> str:
    """Handle Q&A requests during revision sessions"""
    prompt = revision_prompts.QA_RESPONSE_TEMPLATE.format(
        user_question=user_question, current_concept=current_concept,
        content=content, conversation_history=conversation_history
    )
    resp = llm_wrapper.generate_response([{"role":"user","content": prompt}])
    return resp.strip()


@tool
def check_question_relevance(user_input: str, current_concept: str, content: str) -> str:
    """Check if user's question is relevant to current concept"""
    prompt = revision_prompts.RELEVANCE_CHECK_TEMPLATE.format(
        user_input=user_input, current_concept=current_concept, content=content
    )
    resp = llm_wrapper.generate_response([{"role":"user","content": prompt}])
    classification = resp.strip().upper()
    return "RELEVANT" if "RELEVANT" in classification else "IRRELEVANT"


@tool
def handle_custom_input(user_input: str, current_concept: str, content: str, 
                       conversation_history: str = "") -> str:
    """Handle custom/irrelevant user input"""
    relevance = check_question_relevance.invoke({
        "user_input": user_input,
        "current_concept": current_concept,
        "content": content
    })
    
    if relevance == "RELEVANT":
        return handle_qa_request.invoke({
            "user_question": user_input,
            "current_concept": current_concept,
            "content": content,
            "conversation_history": conversation_history
        })
    else:
        prompt = revision_prompts.CUSTOM_INPUT_TEMPLATE.format(
            user_input=user_input, current_concept=current_concept,
            content=content, conversation_history=conversation_history
        )
        resp = llm_wrapper.generate_response([{"role":"user","content": prompt}])
        return resp.strip()


@tool
def detect_question_intent(user_input: str, current_concept: str, 
                          conversation_history: str = "") -> str:
    """Detect if user input is a question or answer"""
    prompt = revision_prompts.QUESTION_DETECTION_TEMPLATE.format(
        user_input=user_input, current_concept=current_concept,
        conversation_history=conversation_history
    )
    resp = llm_wrapper.generate_response([{"role":"user","content": prompt}])
    classification = resp.strip().upper()
    
    if "ASKING_QUESTION" in classification:
        return "ASKING_QUESTION"
    elif "PROVIDING_ANSWER" in classification:
        return "PROVIDING_ANSWER"
    else:
        return "PROVIDING_ANSWER"


# Class for backward compatibility
class RevisionAgent:
    def __init__(self, llm=None):
        self.llm = llm or llm_wrapper
        self.generate_structured_explanation = generate_structured_explanation
        self.generate_explanation_steps = generate_explanation_steps
        self.generate_examples = generate_examples
        self.make_check_question = make_check_question
        self.extract_expected_keywords = extract_expected_keywords
        self.evaluate_answer = evaluate_answer
        self.handle_qa_request = handle_qa_request
        self.check_question_relevance = check_question_relevance
        self.handle_custom_input = handle_custom_input
        self.detect_question_intent = detect_question_intent

    def get_all_tools(self) -> List:
        """Get all tools as a list for LangGraph integration"""
        return [
            generate_structured_explanation,
            generate_explanation_steps,
            generate_examples,
            make_check_question,
            extract_expected_keywords,
            evaluate_answer,
            handle_qa_request,
            check_question_relevance,
            handle_custom_input,
            detect_question_intent
        ]