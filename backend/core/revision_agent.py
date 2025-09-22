from typing import List, Dict, Any
from .llm import GeminiLLMWrapper
from backend.prompts import revision_prompts
import asyncio

llm_wrapper = GeminiLLMWrapper()

class RevisionAgent:
    def __init__(self, llm=None):
        self.llm = llm or llm_wrapper

    async def generate_structured_explanation(self, title: str, content: str, conversation_history: str = "") -> List[Dict[str, Any]]:
        """Generate structured explanation with multiple detailed bubbles"""
        prompt = revision_prompts.STRUCTURED_EXPLANATION_TEMPLATE.format(
            title=title, 
            content=content, 
            conversation_history=conversation_history
        )
        
        resp = await self.llm.generate_response([{"role":"user","content": prompt}])
        
        # Parse the structured response into separate bubbles
        bubbles = self._parse_detailed_bubbles(resp, title)
        return bubbles

    def _parse_detailed_bubbles(self, response: str, title: str) -> List[Dict[str, Any]]:
        """Parse LLM response into detailed structured bubbles"""
        bubbles = []
        
        # Split by bubble headers
        sections = {}
        current_section = None
        current_content = []
        
        lines = response.split('\n')
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check for bubble headers
            if line_stripped.startswith('BUBBLE_1_DEFINITION:'):
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'definition'
                current_content = []
            elif line_stripped.startswith('BUBBLE_2_TECHNICAL:'):
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'technical'
                current_content = []
            elif line_stripped.startswith('BUBBLE_3_EXAMPLES:'):
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'examples'
                current_content = []
            elif line_stripped and current_section:
                # Add content to current section
                current_content.append(line)
        
        # Add the last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        # Create bubbles from sections
        if 'definition' in sections:
            bubbles.append({
                "assistant_message": f"💬 **Bubble 1 – Definition**\n\n{sections['definition'].strip()}",
                "message_type": "concept_section",
                "section": "Definition"
            })
        
        if 'technical' in sections:
            bubbles.append({
                "assistant_message": f"💬 **Bubble 2 – Beginner Technical Explanation**\n\n{sections['technical'].strip()}",
                "message_type": "concept_section",
                "section": "Technical"
            })
        
        if 'examples' in sections:
            bubbles.append({
                "assistant_message": f"💬 **Bubble 3 – Examples**\n\n{sections['examples'].strip()}",
                "message_type": "concept_section",
                "section": "Examples"
            })
        
        # Fallback if parsing fails
        if not bubbles:
            bubbles = self._create_detailed_fallback(response, title)
        
        return bubbles

    def _create_detailed_fallback(self, response: str, title: str) -> List[Dict[str, Any]]:
        """Create detailed fallback structure if parsing fails"""
        # Split response into reasonable chunks
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        
        bubbles = []
        
        # Definition bubble
        definition_content = f"💬 **Bubble 1 – Definition**\n\n**{title}:** "
        if paragraphs:
            definition_content += paragraphs[0]
        else:
            definition_content += "An important concept in this topic that helps us understand how things work."
        
        bubbles.append({
            "assistant_message": definition_content,
            "message_type": "concept_section",
            "section": "Definition"
        })
        
        # Technical explanation bubble
        if len(paragraphs) > 1:
            technical_content = f"💬 **Bubble 2 – Beginner Technical Explanation**\n\n🔬 **{title}** {paragraphs[1]}"
            bubbles.append({
                "assistant_message": technical_content,
                "message_type": "concept_section",
                "section": "Technical"
            })
        
        # Examples bubble
        examples_content = f"💬 **Bubble 3 – Examples**\n\n"
        if len(paragraphs) > 2:
            examples_content += paragraphs[2]
        else:
            examples_content += f"* Example 1 related to **{title.lower()}**.\n* Example 2 showing **{title.lower()}** in action."
        
        bubbles.append({
            "assistant_message": examples_content,
            "message_type": "concept_section",
            "section": "Examples"
        })
        
        return bubbles

    # Keep all other existing methods unchanged
    async def generate_explanation_steps(self, title: str, content: str, conversation_history: str = "", steps: int = 3) -> List[str]:
        """Legacy method for backward compatibility"""
        prompt = revision_prompts.EXPLANATION_TEMPLATE.format(
            title=title, steps=steps, content=content, conversation_history=conversation_history
        )
        resp = await self.llm.generate_response([{"role":"user","content": prompt}])
        lines = [l.strip() for l in resp.splitlines() if l.strip()]
        if len(lines) == 0:
            parts = [s.strip() for s in resp.replace("\n", " ").split(".") if s.strip()]
            lines = [f"{i+1}. {parts[i]}." for i in range(min(steps, len(parts)))]
        return lines[:steps]

    async def generate_examples(self, title: str, content: str, conversation_history: str = "") -> str:
        """Generate practical examples for the given concept"""
        prompt = revision_prompts.EXAMPLES_TEMPLATE.format(
            title=title, 
            content=content, 
            conversation_history=conversation_history
        )
        resp = await self.llm.generate_response([{"role":"user","content": prompt}])
        return resp.strip()

    async def make_check_question(self, title: str, content: str, conversation_history: str = "") -> str:
        prompt = revision_prompts.CHECK_QUESTION_TEMPLATE.format(
            title=title, content=content, conversation_history=conversation_history
        )
        resp = await self.llm.generate_response([{"role":"user","content": prompt}])
        return resp.strip()

    async def extract_expected_keywords(self, title: str, content: str, question: str) -> List[str]:
        """Ask LLM to return minimal keywords for marking correctness."""
        prompt = revision_prompts.KEYWORDS_EXTRACTION_TEMPLATE.format(
            title=title,
            content=content,
            question=question
        )
        resp = await self.llm.generate_response([{"role":"user","content": prompt}])
        text = resp.strip()
        try:
            import json
            data = json.loads(text)
            if isinstance(data, list):
                return [str(x).strip().lower() for x in data if str(x).strip()]
        except Exception:
            pass
        return [w.lower() for w in title.split()[:3]]

    async def evaluate_answer(self, user_answer: str, expected_keywords: List[str], conversation_history: str = "", *, title: str = "", content: str = "", assistant_message: str = "", check_question: str = "") -> Dict[str, Any]:
        """Evaluate using full context instead of relying on keyword matches."""
        prompt = revision_prompts.EVAL_WITH_CONTEXT_TEMPLATE.format(
            title=title,
            content=content,
            assistant_message=assistant_message,
            check_question=check_question,
            user_answer=user_answer
        )
        resp = await self.llm.generate_response([{"role":"user","content": prompt}])
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
        return {"verdict": verdict, "justification": justification, "correction": correction or justification}

    async def handle_qa_request(self, user_question: str, current_concept: str, content: str, conversation_history: str = "") -> str:
        """Handle Q&A requests during revision sessions"""
        prompt = revision_prompts.QA_RESPONSE_TEMPLATE.format(
            user_question=user_question,
            current_concept=current_concept,
            content=content,
            conversation_history=conversation_history
        )
        resp = await self.llm.generate_response([{"role":"user","content": prompt}])
        return resp.strip()

    async def check_question_relevance(self, user_input: str, current_concept: str, content: str) -> str:
        """Check if the user's question is relevant to the current concept"""
        prompt = revision_prompts.RELEVANCE_CHECK_TEMPLATE.format(
            user_input=user_input,
            current_concept=current_concept,
            content=content
        )
        resp = await self.llm.generate_response([{"role":"user","content": prompt}])
        classification = resp.strip().upper()
        if "RELEVANT" in classification:
            return "RELEVANT"
        else:
            return "IRRELEVANT"

    async def handle_custom_input(self, user_input: str, current_concept: str, content: str, conversation_history: str = "") -> str:
        """Handle custom/irrelevant user input during revision sessions"""
        relevance = await self.check_question_relevance(user_input, current_concept, content)
        
        if relevance == "RELEVANT":
            return await self.handle_qa_request(user_input, current_concept, content, conversation_history)
        else:
            prompt = revision_prompts.CUSTOM_INPUT_TEMPLATE.format(
                user_input=user_input,
                current_concept=current_concept,
                content=content,
                conversation_history=conversation_history
            )
            resp = await self.llm.generate_response([{"role":"user","content": prompt}])
            return resp.strip()

    async def detect_question_intent(self, user_input: str, current_concept: str, conversation_history: str = "") -> str:
        """Dynamically detect if user input is a question or an answer using LLM"""
        prompt = revision_prompts.QUESTION_DETECTION_TEMPLATE.format(
            user_input=user_input,
            current_concept=current_concept,
            conversation_history=conversation_history
        )
        resp = await self.llm.generate_response([{"role":"user","content": prompt}])
        classification = resp.strip().upper()
        if "ASKING_QUESTION" in classification:
            return "ASKING_QUESTION"
        elif "PROVIDING_ANSWER" in classification:
            return "PROVIDING_ANSWER"
        else:
            return "PROVIDING_ANSWER"