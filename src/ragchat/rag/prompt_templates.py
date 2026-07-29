"""Prompt templates for the RAG chatbot pipeline.

All prompts are versioned here so that changes to LLM instructions
can be reviewed and tracked in one place.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ---------------------------------------------------------------------------
# 1. CASUAL CONVERSATION
# ---------------------------------------------------------------------------

CASUAL_SYSTEM = (
    "You are a friendly, intelligent, and helpful AI assistant. "
    "You can answer general knowledge questions, engage in small talk, "
    "explain concepts, and assist with a wide range of topics. "
    "Format your responses cleanly with distinct line breaks, bold text, and bullet points where helpful."
)

CASUAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CASUAL_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}"),
])


# ---------------------------------------------------------------------------
# 2. UNIVERSAL QUERY ROUTING / DECISION
# ---------------------------------------------------------------------------

DECISION_PROMPT_TEMPLATE = (
    "You are a universal query router for an enterprise RAG assistant.\n\n"
    "Categorize the user's question into EXACTLY ONE of three categories:\n\n"
    "Answer RETRIEVE if:\n"
    "  - The question is about company details, services, team, projects, statistics, architecture, or any facts covered in uploaded documents or knowledge base\n"
    "  - The user asks about specific organization information, technical specifications, or uploaded PDF documents\n\n"
    "Answer WEB if:\n"
    "  - The question is about real-time live facts, current news, live stock prices, or recent world events\n"
    "  - The question explicitly requests real-time web search information\n\n"
    "Answer RESPOND if:\n"
    "  - The question is a greeting or small talk (e.g. 'hi', 'hello', 'how are you')\n"
    "  - The question is about general math or simple conversational queries\n"
    "  - The question is about the assistant itself\n\n"
    "User question: \"{question}\"\n\n"
    "Answer with EXACTLY ONE WORD — RETRIEVE, WEB, or RESPOND:"
)


# ---------------------------------------------------------------------------
# 3. QUERY REPHRASING
# ---------------------------------------------------------------------------

REPHRASE_SYSTEM = (
    "You are a query rephrasing assistant for search retrieval.\n"
    "Given a conversation history and the latest user message, rewrite the "
    "latest message as a short 1-sentence standalone search query.\n\n"
    "CRITICAL RULES:\n"
    "1. Output ONLY the short 1-sentence search query (MAXIMUM 15 WORDS).\n"
    "2. DO NOT answer the question. DO NOT provide explanations, bullet points, or code.\n"
    "3. Resolve pronouns (e.g. 'it', 'they', 'them') using the conversation history.\n"
    "4. If the question is already clear, return it word-for-word as-is."
)

REPHRASE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", REPHRASE_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}"),
])


# ---------------------------------------------------------------------------
# 4. QUERY EXPANSION
# ---------------------------------------------------------------------------

QUERY_EXPANSION_SYSTEM = (
    "You are an expert search assistant. "
    "Your task is to generate 2 alternative phrasings or search query variations "
    "for the user's input question to help retrieve the most relevant sections "
    "from the document corpus.\n\n"
    "Look at the underlying semantic intent or meaning of the question and "
    "produce variations that explore different angles.\n\n"
    "Respond with ONLY the variations, one per line. "
    "Do NOT add numbering, bullets, or any introduction."
)

QUERY_EXPANSION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QUERY_EXPANSION_SYSTEM),
    ("user", "Original Query: {query}"),
])


# ---------------------------------------------------------------------------
# 5. DOCUMENT GRADING
# ---------------------------------------------------------------------------

DOCUMENT_GRADER_SYSTEM = (
    "You are a strict relevance grader assessing whether a retrieved document chunk "
    "contains information that is relevant or helpful to answer the user's query.\n\n"
    "If the document contains keywords, facts, or semantic meaning related to the question, "
    "grade it as relevant. Otherwise grade it as irrelevant.\n\n"
    "Respond with a single JSON object with two fields:\n"
    "  - 'binary_score': either 'yes' (relevant) or 'no' (irrelevant)\n"
    "  - 'reasoning': a single sentence explaining your decision"
)

DOCUMENT_GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", DOCUMENT_GRADER_SYSTEM),
    ("user", "User Question: {query}\n\nRetrieved Document Chunk:\n{document}\n\nRelevance score:"),
])


# ---------------------------------------------------------------------------
# 6. UNIVERSAL RAG GENERATION
# ---------------------------------------------------------------------------

GENERATION_SYSTEM = (
    "You are an intelligent, precise, and professional enterprise AI assistant.\n\n"
    "Instructions:\n"
    "1. Synthesize a comprehensive, clear, and accurate answer strictly based on the provided Retrieved Document Context below.\n"
    "2. Structure your response using clean Markdown with distinct paragraph line breaks (\\n\\n), section headers (###), bold key terms, and bullet points.\n"
    "3. Highlight key statistics, metrics, company details, services, or technical steps clearly.\n"
    "4. Maintain a professional, helpful tone and do not hallucinate details beyond the provided context."
)

GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", GENERATION_SYSTEM),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "Retrieved Document Context:\n{context}\n\nUser Question: {query}"),
])
