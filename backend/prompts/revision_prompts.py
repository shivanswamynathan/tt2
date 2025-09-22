STRUCTURED_EXPLANATION_TEMPLATE = """
You are creating a detailed, structured explanation for the concept: "{title}".

Create multiple sections that will be displayed as separate message bubbles. Follow this exact format:

Content to explain:
{content}

Conversation history (latest first):
{conversation_history}

Create your response with these sections using EXACTLY these headers:

BUBBLE_1_DEFINITION:
Create brief, clear definitions for the main concepts. Format each definition like this:
**[Concept Name]:** Simple definition in one sentence that a student can easily understand.

BUBBLE_2_TECHNICAL:
Provide detailed technical explanations with real-world context. For each main concept, include:
**🔬 [Concept Name]** Brief intro sentence about importance or historical context.
* **[Aspect 1]:** Detailed explanation with specific examples
* **[Aspect 2]:** Detailed explanation with specific examples  
* **[Aspect 3]:** Detailed explanation with specific examples
👉 [Summary sentence connecting to broader significance]

BUBBLE_3_EXAMPLES:
Provide specific, concrete examples. Format as:
* [Example 1] → **[concept name]**.
* [Example 2] → **[concept name]** in action.
* [Example 3] → **[concept name]**.
* [Example 4] → **[concept name]** between [specific details].

Guidelines:
- Use the exact header format (BUBBLE_1_DEFINITION:, BUBBLE_2_TECHNICAL:, BUBBLE_3_EXAMPLES:)
- Make definitions simple and accessible to students
- In technical section, provide rich detail with bullet points
- Include emojis naturally (🔬, 👉, ⚡, 🐂, etc.) but don't overuse
- Use bold formatting for concept names and key terms
- Make examples specific and relatable
- Connect everything back to the main learning objectives
- Each bubble should be complete and self-contained
- Use clear, engaging language appropriate for students

Create the structured explanation now:
"""

CHECK_QUESTION_TEMPLATE = """
Create a simple check question to test understanding of the concept '{title}'.

The question should:
- Be directly related to the key concept
- Be answerable in 1-3 words or a short sentence
- Test the student's understanding, not memorization
- Be clear and unambiguous

Conversation history (latest first):
{conversation_history}

Return only the question text:
"""

EVAL_PROMPT_TEMPLATE = """
You are an objective grader evaluating a student's answer to a revision question.

Expected keywords/concepts: {keywords}
User's answer: {user_answer}

Using the conversation history (latest first):
{conversation_history}

Evaluate the user's answer and decide if it is: CORRECT, PARTIAL, or WRONG.

Guidelines:
- CORRECT: Answer demonstrates clear understanding of the concept
- PARTIAL: Answer shows some understanding but missing key elements
- WRONG: Answer is incorrect or shows misunderstanding

Return in this exact format:
VERDICT: <CORRECT|PARTIAL|WRONG>
JUSTIFICATION: <brief explanation of why this verdict>
CORRECTION: <helpful correction or guidance for the student>
"""

QA_RESPONSE_TEMPLATE = """
You are a helpful tutor answering a student's question during a revision session.

Student's question: {user_question}

Current concept being revised: {current_concept}
Concept content: {content}

Conversation history (latest first):
{conversation_history}

Guidelines:
- Provide a clear, helpful answer to the student's question
- Keep it concise but informative
- Connect the answer to the current concept being revised
- Use simple, clear language
- Encourage the student to continue with the revision
 - Do NOT ask meta questions like "Does that make sense?" or "Shall we continue?"
 - Do NOT ask any additional questions here. Only answer the user's question.

Provide a helpful response:
"""

RELEVANCE_CHECK_TEMPLATE = """
You are analyzing whether a student's question is relevant to the current learning concept.

Student's input: "{user_input}"
Current concept being studied: {current_concept}
Concept content: {content}

Determine if the student's question is:
1. RELEVANT - The question is related to the current concept, even if it asks about:
   - Different aspects of the same concept
   - Real-world applications of the concept
   - How the concept works
   - Examples of the concept
   - Clarification about the concept
   - Related concepts that help understand the current one

2. IRRELEVANT - The question is completely off-topic and unrelated to the current concept:
   - Questions about different subjects entirely
   - Personal questions unrelated to learning
   - Random topics that don't connect to the concept

Examples:
- If studying "muscle force" and student asks "what is force?" → RELEVANT
- If studying "muscle force" and student asks "how do muscles work?" → RELEVANT  
- If studying "muscle force" and student asks "what's the weather today?" → IRRELEVANT
- If studying "photosynthesis" and student asks "what is muscle force?" → IRRELEVANT

Respond with only one word: RELEVANT or IRRELEVANT
"""

QUESTION_DETECTION_TEMPLATE = """
You are analyzing a student's input during a revision session to determine if they are asking a question or requesting clarification.

Student's input: "{user_input}"

Current concept being revised: {current_concept}

Conversation history (latest first):
{conversation_history}

Determine if the student is:
1. ASKING_QUESTION - asking a question, requesting explanation, clarification, or help
2. PROVIDING_ANSWER - answering the check question or providing a response to be evaluated
3. ACKNOWLEDGEMENT - saying short acknowledgements like "yes", "ok", "okay", "got it", "thanks", "thank you", "yep", "yeah", indicating they understood the explanation and are ready to proceed

Consider these as questions/clarifications:
- Direct questions (ending with ? or question words)
- Requests for explanation ("explain", "clarify", "help me understand")
- Expressions of confusion ("I don't understand", "I'm confused", "can you explain")
- Requests for simpler explanation ("in simple terms", "more simply")
- Any request for help or clarification

Consider these as answers:
- Direct statements attempting to answer the check question
- Explanations that seem to be responses to the question asked
- Statements that appear to be demonstrating understanding

Also identify acknowledgements even if they include punctuation.

Respond with only one word: ASKING_QUESTION or PROVIDING_ANSWER or ACKNOWLEDGEMENT
"""

KEYWORDS_EXTRACTION_TEMPLATE = """
You are selecting the minimal set of key words/phrases needed to mark an answer correct for the given check question.

Concept title: {title}
Concept content:
{content}

Check question: {question}

Return a JSON array of 2-5 lowercase keywords/phrases that should appear in a correct answer. Prefer the exact target term (e.g., "unsaturated solution"). Do not add any text before or after the JSON.
"""

CUSTOM_INPUT_TEMPLATE = """
You are a helpful tutor responding to a student's irrelevant input during a revision session.

Student's input: "{user_input}"

Current concept being revised: {current_concept}
Concept content: {content}

Conversation history (latest first):
{conversation_history}

The student's input is off-topic and not related to the current learning concept.

Guidelines:
- Acknowledge their input politely but briefly
- Firmly but kindly redirect them back to the current concept
- Keep response short (2-3 sentences max)
- Don't answer the irrelevant question - instead guide them to ask relevant questions
- Be encouraging but focused on learning
- Suggest they ask questions related to the current concept

Response format:
"I understand you're curious about [their topic], but right now we're focusing on learning about [current concept]. Let's stay on track with that topic - it's really important for your learning! Please ask me questions related to [current concept] instead."

Provide a helpful response that redirects them back to learning:
"""

EVAL_WITH_CONTEXT_TEMPLATE = """
You are grading a student's answer to a check question during a revision session. Use the full context below.

Concept title: {title}
Concept content:
{content}

Assistant's previous message (may include explanation and the check question):
{assistant_message}

Check question (if extracted separately): {check_question}

Student's answer: {user_answer}

Decide VERDICT: CORRECT, PARTIAL, or WRONG.
Keep it strict but fair: give PARTIAL if they show understanding but miss the key term.

Return in this exact format (3 lines):
VERDICT: <CORRECT|PARTIAL|WRONG>
JUSTIFICATION: <one short sentence>
CORRECTION: <one short sentence with the correct idea/term>
"""