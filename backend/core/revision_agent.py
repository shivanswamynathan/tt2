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
    
    # Enhanced prompt with strict formatting requirements
    enhanced_prompt = f"""{revision_prompts.STRUCTURED_EXPLANATION_TEMPLATE.format(
        title=title, content=content, conversation_history=conversation_history
    )}

CRITICAL FORMATTING REQUIREMENTS:
1. Start each bubble with the EXACT header: BUBBLE_1_DEFINITION:, BUBBLE_2_TECHNICAL:, BUBBLE_3_EXAMPLES:
2. Each bubble must be comprehensive and detailed (minimum 100 words each)
3. Include ALL the content structure shown in the template
4. Do NOT skip any sections or bubbles
5. Use proper markdown formatting with ** for bold text
6. Include emojis as shown in the template

Example structure:
BUBBLE_1_DEFINITION:
💡 **Core Concept**
[Full content here...]

BUBBLE_2_TECHNICAL:
🔬 **How It Works**
[Full content here...]

BUBBLE_3_EXAMPLES:
📚 **Examples & Applications**
[Full content here...]
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
    """Parse LLM response into detailed structured bubbles - PRESERVES ALL CONTENT"""
    bubbles = []
    sections = {}
    current_section = None
    current_content = []
    
    lines = response.split('\n')
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check for bubble headers
        if line_stripped.startswith('BUBBLE_1_DEFINITION:'):
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = 'definition'
            current_content = []
            continue  # Skip the header line
            
        elif line_stripped.startswith('BUBBLE_2_TECHNICAL:'):
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = 'technical'
            current_content = []
            continue
            
        elif line_stripped.startswith('BUBBLE_3_EXAMPLES:'):
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = 'examples'
            current_content = []
            continue
        
        # Add content to current section (preserve ALL lines including blank ones)
        if current_section is not None:
            current_content.append(line)
    
    # Don't forget the last section
    if current_section and current_content:
        sections[current_section] = '\n'.join(current_content).strip()
    
    logger.info(f"Parsed sections: {list(sections.keys())}")
    for section_name, section_content in sections.items():
        logger.info(f"  {section_name}: {len(section_content)} characters")
    
    # Create bubbles from sections - NO TRUNCATION
    if 'definition' in sections and sections['definition']:
        bubbles.append({
            "assistant_message": sections['definition'],
            "message_type": "concept_section",
            "section": "Definition"
        })
    
    if 'technical' in sections and sections['technical']:
        bubbles.append({
            "assistant_message": sections['technical'],
            "message_type": "concept_section",
            "section": "Technical"
        })
        
    if 'examples' in sections and sections['examples']:
        bubbles.append({
            "assistant_message": sections['examples'],
            "message_type": "concept_section",
            "section": "Examples"
        })
    
    # Fallback if parsing failed (less than 3 bubbles or any bubble too short)
    if len(bubbles) < 3 or any(len(b['assistant_message']) < 50 for b in bubbles):
        logger.warning(f"Parsing failed or produced insufficient content. Bubbles: {len(bubbles)}")
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
    
    # STRATEGY 1: Split by paragraphs (double newline)
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    # STRATEGY 2: If not enough paragraphs, split by single newline
    if len(paragraphs) < 3:
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
    
    # STRATEGY 3: If still not enough, split by sentences
    if len(paragraphs) < 3:
        import re
        sentences = re.split(r'[.!?]\s+', content)
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        # Group sentences into chunks of 2-3
        paragraphs = []
        i = 0
        while i < len(sentences):
            chunk_size = min(3, len(sentences) - i)
            chunk = ' '.join(sentences[i:i+chunk_size])
            paragraphs.append(chunk)
            i += chunk_size
    
    logger.info(f"Split content into {len(paragraphs)} sections")
    
    # STRATEGY 4: If we STILL don't have enough, divide the content into thirds
    if len(paragraphs) < 3:
        content_length = len(content)
        third = content_length // 3
        
        def find_break_point(text, target_pos):
            """Find a good break point (sentence end) near target position"""
            # Look for sentence endings within 100 chars of target
            search_start = max(0, target_pos - 50)
            search_end = min(len(text), target_pos + 50)
            
            for i in range(target_pos, search_end):
                if i < len(text) and text[i] in '.!?\n':
                    return i + 1
            
            # If no break found ahead, look backwards
            for i in range(target_pos, search_start, -1):
                if i < len(text) and text[i] in '.!?\n':
                    return i + 1
            
            return target_pos
        
        break1 = find_break_point(content, third)
        break2 = find_break_point(content, 2 * third)
        
        paragraphs = [
            content[:break1].strip(),
            content[break1:break2].strip(),
            content[break2:].strip()
        ]
        paragraphs = [p for p in paragraphs if p]
    
    # Ensure minimum of 3 sections
    while len(paragraphs) < 3 and paragraphs:
        # Split the longest paragraph
        longest_idx = max(range(len(paragraphs)), key=lambda i: len(paragraphs[i]))
        longest = paragraphs[longest_idx]
        
        if len(longest) > 100:
            mid = len(longest) // 2
            # Find sentence break near middle
            for i in range(mid, min(mid + 50, len(longest))):
                if longest[i] in '.!?':
                    part1 = longest[:i+1].strip()
                    part2 = longest[i+1:].strip()
                    paragraphs[longest_idx:longest_idx+1] = [part1, part2]
                    break
            else:
                break
        else:
            break
    
    # BUILD BUBBLE 1: Definition - Use first section(s)
    definition_sections = paragraphs[:max(1, len(paragraphs)//3)]
    definition_content = f"💡 **Definition & Core Concepts**\n\n**{title}:**\n\n"
    definition_content += '\n\n'.join(definition_sections)
    
    bubbles.append({
        "assistant_message": definition_content,
        "message_type": "concept_section",
        "section": "Definition"
    })
    
    # BUILD BUBBLE 2: Technical - Use middle section(s)
    mid_start = max(1, len(paragraphs)//3)
    mid_end = max(2, 2*len(paragraphs)//3)
    technical_sections = paragraphs[mid_start:mid_end]
    
    if technical_sections:
        technical_content = f"🔬 **How It Works**\n\n"
        technical_content += '\n\n'.join(technical_sections)
    else:
        technical_content = f"🔬 **How It Works**\n\n{paragraphs[1] if len(paragraphs) > 1 else content}"
    
    bubbles.append({
        "assistant_message": technical_content,
        "message_type": "concept_section",
        "section": "Technical"
    })
    
    # BUILD BUBBLE 3: Examples - Use last section(s)
    examples_start = max(2, 2*len(paragraphs)//3)
    examples_sections = paragraphs[examples_start:]
    
    if examples_sections:
        examples_content = f"📚 **Examples & Applications**\n\n"
        examples_content += '\n\n'.join(examples_sections)
    else:
        examples_content = f"📚 **Examples & Applications**\n\n"
        examples_content += paragraphs[-1] if paragraphs else content
    
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
    return [
        {
            "assistant_message": f"💡 **Definition & Core Concepts**\n\n**{title}:** An important concept in this topic that helps us understand fundamental principles.",
            "message_type": "concept_section",
            "section": "Definition"
        },
        {
            "assistant_message": f"🔬 **How It Works**\n\n**{title}** involves understanding the key mechanisms and processes that make this concept work in practice.",
            "message_type": "concept_section",
            "section": "Technical"
        },
        {
            "assistant_message": f"📚 **Examples & Applications**\n\n🏠 **Example 1:** Everyday applications of {title.lower()}\n\n🔬 **Example 2:** Scientific demonstrations of {title.lower()}",
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