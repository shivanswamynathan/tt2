STRUCTURED_EXPLANATION_TEMPLATE = """
Create a structured explanation for: "{title}"

Content: {content}
History: {conversation_history}

Generate EXACTLY 3 separate messages/bubbles separated by "|||":

MESSAGE 1: Concept Name Only (Maximum 20 characters)
💡 **[Write ONLY the concept name in 1-3 words, maximum 20 characters]**

|||

MESSAGE 2: Full Explanation
📖 **What is [concept]?**

[Provide a clear, detailed explanation of the concept. Include:
- What it is (definition)
- How it works (process/mechanism)  
- Why it's important
- Key characteristics or properties
Use 3-5 paragraphs with proper formatting and emojis for clarity]

|||

MESSAGE 3: Examples
🌟 **Examples of [concept]**

[Provide 2-3 concrete, real-world examples. Each example should:
- Be relatable and easy to understand
- Connect directly to the concept
- Include specific details
Format each example with emoji markers (🏠, 🌍, 🔬, etc.)]

CRITICAL REQUIREMENTS:
✓ MESSAGE 1: ONLY concept name - MAXIMUM 20 CHARACTERS
✓ MESSAGE 2: Full detailed explanation (100+ words)
✓ MESSAGE 3: 2-3 concrete examples with details
✓ Separate each message with "|||"
✓ Do NOT include labels like "BUBBLE_1", "MESSAGE 1", etc. in the output
✓ Use emojis and bold text for visual organization
✓ Student-friendly language throughout
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

BE EXTREMELY STRICT. This is a focused learning session, NOT a general conversation.

Classify as RELEVANT ONLY if:
- The question directly asks about the concept itself (definitions, explanations)
- The question asks for clarification about what was just taught
- The question asks for examples of THIS EXACT concept
- The question uses technical terms from the current content

Classify as IRRELEVANT if:
- The question mentions people, celebrities, actors, or public figures
- The question mentions brands, products, movies, or entertainment
- The question is about real-world applications NOT mentioned in the content
- The question requires general knowledge beyond the concept
- The question is about "who", "when", "where" (trivia) rather than "what", "how", "why" (concept understanding)
- The question seems like curiosity or conversation rather than learning

CRITICAL EXAMPLES:
- Current: "Forces and Motion" | Question: "what is applied force?" → RELEVANT
- Current: "Forces and Motion" | Question: "how does friction work?" → RELEVANT  
- Current: "Forces and Motion" | Question: "who is Ajith Kumar?" → IRRELEVANT (person)
- Current: "Forces and Motion" | Question: "who is ultimate star?" → IRRELEVANT (celebrity)
- Current: "Forces and Motion" | Question: "what car does Ajith drive?" → IRRELEVANT (trivia)
- Current: "Forces and Motion" | Question: "how do race cars work?" → IRRELEVANT (not in content)
- Current: "Photosynthesis" | Question: "what is chlorophyll?" → RELEVANT
- Current: "Photosynthesis" | Question: "who discovered photosynthesis?" → IRRELEVANT (history trivia)

THE RULE: If the question can be answered WITHOUT the current concept content, it's IRRELEVANT.

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
EXAMPLES_TEMPLATE = """
# Identity

You are an educational assistant that provides practical, concrete examples to help students understand the concept: "{title}".

# Instructions

## STRICT EXAMPLE COUNT RULE
⚠️ You MUST output **exactly {num_examples} examples**.  
* If content length is ≤ 100 characters → give exactly 3 examples.  
* If content length is > 100 characters → give exactly 2 examples.  
Never provide more or fewer than the required number.  
STOP after the final required example. Do not add extra examples.  

## Core Requirements
* Only provide examples — no definitions or structured explanations.
* Use clear, simple language appropriate for students.
* Make examples relatable to daily life where possible.
* Connect each example back to the main concept.
* Include real-world applications when relevant.
* Use formatting to make examples easy to scan and understand.

## Example Structure
* For 2 examples → Each should be detailed and thorough.
* For 3 examples → Each should be simple, clear, and focused.
* Use emoji markers (🌟) before each example.
* Use **bold** for the concept title inside explanations.
* Include arrow (→) before explanations.
* End with a **single key takeaway** sentence.

# Context

📝 Current concept content: {content}  
📚 Conversation history (latest first): {conversation_history}  
📏 Content length: {content_length} characters  
✅ Required number of examples: {num_examples}  

# Output Format

**Here are some practical examples to help you understand {title} better:**

🌟 **Example 1:** [Scenario]  
→ This shows **{title}** because [explanation]

🌟 **Example 2:** [Scenario]  
→ This demonstrates **{title}** because [explanation]

{example_3}

💡 **Key takeaway:** [One sentence that ties all examples together]

# Task

CRITICAL: You must provide **exactly {num_examples} examples** and then stop.
"""

EXPLANATION_TEMPLATE = """
# ENHANCED EXPLANATION TEMPLATE
🎓 You are explaining the concept '{title}' to a student in {steps} clear, progressive steps.

📝 Content to explain:
{content}

📚 Conversation history (latest first):
{conversation_history}

🎯 Guidelines:
- 🔢 Break down the concept into {steps} logical, sequential steps
- 🏗️ Start with the most basic understanding and build complexity gradually  
- 💬 Use simple, clear language appropriate for students
- ⬆️ Each step should build on the previous one logically
- 🌟 Include concrete, relatable examples in each step
- 📖 Make each step elaborate and comprehensive, not just brief sentences
- 🎪 Use active voice and engaging, enthusiastic language
- 🧩 Connect each step to real-world applications when possible
- 💡 Add visual emojis to make explanations more engaging
- 🔍 Provide detailed explanations with "why" and "how" context
- 📊 Include analogies, comparisons, or metaphors where helpful

📋 Format Requirements:
- Use exactly {steps} numbered steps
- Format each step as: "🔸 **Step X:** [Elaborate explanation with examples, context, and details]"
- Each step should be 2-4 sentences minimum, providing thorough understanding
- Include relevant emojis throughout each step explanation
- Bold key terms and concepts within each step
- End each step with a practical example or application when possible

🚀 Provide the detailed step-by-step explanation now:
"""