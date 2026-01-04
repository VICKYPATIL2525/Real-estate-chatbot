import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from dotenv import load_dotenv
load_dotenv()

# ------------------------------
# CONFIG
# ------------------------------
PERSIST_DIRECTORY = "./chroma_db"

AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = "https://inadev-saraswati-openai.openai.azure.com/"
AZURE_VERSION = "2024-12-01-preview"
AZURE_DEPLOYMENT = "gpt-4.1-mini"


# ------------------------------
# SAFE LOAD FUNCTIONS
# ------------------------------
@st.cache_resource
def safe_load_embeddings():
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except Exception as e:
        st.error(f"❌ Embedding load failed: {e}")
        return None


@st.cache_resource
def safe_load_db(embeddings):
    try:
        return Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
    except Exception as e:
        st.error(f"❌ Chroma DB load failed: {e}")
        return None


@st.cache_resource
def safe_load_llm():
    return AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_VERSION,
        deployment_name=AZURE_DEPLOYMENT,
        temperature=0.1,
        max_tokens=2000
    )


# ------------------------------------------------
# VALID PROJECT → URL DICTIONARY
# ------------------------------------------------
VALID_URLS_DICT = {
    "Mahindra Citadel": "https://www.mahindralifespaces.com/real-estate-properties/pune-property/mahindra-citadel/",
    "Mahindra Happinest Tathawade": "https://www.mahindralifespaces.com/real-estate-properties/pune-property/happinest-tathawade/",
    "Mahindra Aqualily": "https://www.mahindralifespaces.com/real-estate-properties/chennai-property/aqualily-mahindra-world-city-chennai/",
    "Mahindra Happinest Avadi": "https://www.mahindralifespaces.com/real-estate-properties/chennai-property/happinest-avadi/",
    "Mahindra Aura": "https://www.mahindralifespaces.com/real-estate-properties/gurgaon-property/mahindra-aura/",
    "Mahindra NewHaven": "https://mahindralifespacesupcoming.com/bangalore/mahindra-newhaven/",
    "Mahindra Happinest Palghar": "https://www.mahindralifespaces.com/real-estate-properties/mumbai-property/happinest-palghar/",
    "Mahindra Happinest Boisar": "https://www.mahindralifespaces.com/real-estate-properties/mumbai-property/happinest-boisar/",
    "Mahindra Meridian": "https://www.mahindralifespaces.com/real-estate-properties/alibaug-property/mahindra-meridian/",
    "Green Estates by Mahindra": "https://www.mahindralifespaces.com/real-estate-properties/chennai-plots/green-estates-by-mahindra/",
    "Lakefront Estates": "https://www.mahindralifespaces.com/real-estate-properties/chennai-plots/lakefront-estates-by-mahindra/",
    "Origins by Mahindra – Ahmedabad": "https://www.mahindralifespaces.com/origins-by-mahindra/ahmedabad/",
    "Origins by Mahindra – Chennai": "https://www.mahindralifespaces.com/origins-by-mahindra/chennai/",
}


# ------------------------------
# ENHANCED RETRIEVAL
# ------------------------------
def retrieve_chunks_enhanced(query, db, k=15):
    """
    Enhanced retrieval with MMR for better diversity and relevance
    """
    try:
        # Use MMR (Maximal Marginal Relevance) for diverse yet relevant results
        docs = db.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=30,  # Fetch more candidates
            lambda_mult=0.7  # Balance between relevance (1.0) and diversity (0.0)
        )

        # Fallback to similarity search if MMR fails
        if not docs:
            docs = db.similarity_search(query, k=k)

        return docs
    except Exception as e:
        # Fallback to standard similarity search
        try:
            return db.similarity_search(query, k=k)
        except Exception as e2:
            st.error(f"❌ Retrieval Error: {e2}")
            return []


def deduplicate_and_rank_chunks(docs):
    """
    Remove duplicate or highly similar chunks and rank by relevance
    """
    seen_content = set()
    unique_docs = []

    for doc in docs:
        # Simple deduplication based on content similarity
        content_hash = doc.page_content.strip().lower()[:200]  # First 200 chars
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_docs.append(doc)

    return unique_docs


# ------------------------------
# ENHANCED PROMPT BUILDING
# ------------------------------
def build_enhanced_prompt(query, docs, chat_history):
    import json

    # Deduplicate chunks
    docs = deduplicate_and_rank_chunks(docs)

    # Convert dictionary to readable JSON for the prompt
    valid_url_list = json.dumps(VALID_URLS_DICT, indent=2)

    # -------------------------
    # BUILD STRUCTURED CONTEXT
    # -------------------------
    context_parts = []
    for idx, d in enumerate(docs, 1):
        text = d.page_content.strip()
        # Add metadata if available
        metadata = d.metadata if hasattr(d, 'metadata') else {}
        source = metadata.get('source', 'Unknown')

        context_parts.append(f"[Chunk {idx}]\nSource: {source}\nContent: {text}\n")

    context = "\n---\n".join(context_parts)

    # -------------------------
    # BUILD CONVERSATION HISTORY
    # -------------------------
    history_text = ""
    if chat_history:
        history_text = "\n### Recent Conversation:\n"
        for turn in chat_history[-6:]:  # Last 3 exchanges
            role = "User" if isinstance(turn, HumanMessage) else "Assistant"
            history_text += f"{role}: {turn.content}\n"

    # -------------------------
    # ENHANCED SYSTEM PROMPT
    # -------------------------
    system_prompt = f"""You are an expert assistant for Mahindra Lifespaces real estate projects. Your role is to provide accurate, helpful, and well-formatted information about properties.

═══════════════════════════════════════
🔗 VALID PROJECT URLS (AUTHORITATIVE SOURCE)
═══════════════════════════════════════
{valid_url_list}

═══════════════════════════════════════
📋 RESPONSE GUIDELINES
═══════════════════════════════════════

1. **Information Accuracy**:
   - Use ONLY information from the provided context chunks below
   - If information is not in the context, respond: "I don't have that specific information in my current knowledge base."
   - Never fabricate or guess details about properties

2. **Hyperlink Rules**:
   - When mentioning a project, check if it exists in the VALID PROJECT URLS
   - If found: Format as **[Project Name](URL)**
   - If NOT found: Use plain text **Project Name** (no link)
   - NEVER create or modify URLs
   - Use case-insensitive partial matching (e.g., "Citadel" matches "Mahindra Citadel")

3. **Answer Structure**:
   - Start with a direct answer to the user's question
   - Use clear headings and bullet points for readability
   - For multiple projects: List each with key details
   - Format: **[Project Name](URL)** – Brief description
   - Include relevant details: location, size, amenities, pricing (if available)

4. **Formatting**:
   - Use markdown for clean presentation
   - Use **bold** for project names
   - Use bullet points for features/amenities
   - Use tables for comparing multiple projects
   - Keep responses concise but comprehensive

5. **Context Handling**:
   - Consider the conversation history for context
   - If user asks follow-up questions, maintain coherence
   - Reference previous answers when relevant

6. **Edge Cases**:
   - If query is unclear: Ask for clarification
   - If multiple interpretations exist: Address all possibilities
   - If no relevant chunks: Politely state information is unavailable

═══════════════════════════════════════
💬 CONVERSATION CONTEXT
═══════════════════════════════════════
{history_text}

═══════════════════════════════════════
📚 RETRIEVED INFORMATION
═══════════════════════════════════════
{context}

═══════════════════════════════════════
❓ USER QUESTION
═══════════════════════════════════════
{query}

═══════════════════════════════════════
📝 YOUR TASK
═══════════════════════════════════════
Provide a comprehensive, accurate, and well-formatted answer based solely on the retrieved information above. Apply all hyperlink rules and formatting guidelines.
"""

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]


# ------------------------------
# ENHANCED RAG ANSWER
# ------------------------------
def rag_answer_enhanced(query, db, llm, chat_history):
    """
    Enhanced RAG pipeline with better retrieval and prompt engineering
    """
    # Step 1: Retrieve relevant chunks
    docs = retrieve_chunks_enhanced(query, db, k=15)

    if not docs:
        return "❌ No relevant information found in the database. Please try rephrasing your question."

    # Step 2: Build enhanced prompt
    messages = build_enhanced_prompt(query, docs, chat_history)

    # Step 3: Get LLM response
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"❌ Error generating response: {e}"


# ------------------------------
# STREAMLIT UI
# ------------------------------
def main():
    st.set_page_config(
        page_title="Mahindra Lifespaces Chatbot",
        page_icon="🏢",
        layout="wide"
    )

    st.title("🏢 Mahindra Lifespaces – Intelligent Property Assistant")
    st.caption("Powered by Azure OpenAI + ChromaDB | RAG-Enhanced Responses")

    # Load components
    embeddings = safe_load_embeddings()
    db = safe_load_db(embeddings)
    llm = safe_load_llm()

    if None in (embeddings, db, llm):
        st.error("Failed to load required components. Please check your configuration.")
        st.stop()

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(msg.content, unsafe_allow_html=True)

    # Chat input
    user_query = st.chat_input("Ask me anything about Mahindra properties...")

    if user_query:
        # Add user message to history
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching knowledge base..."):
                answer = rag_answer_enhanced(
                    user_query,
                    db,
                    llm,
                    st.session_state.chat_history
                )
            st.markdown(answer, unsafe_allow_html=True)

        # Add assistant response to history
        st.session_state.chat_history.append(AIMessage(content=answer))

    # Sidebar
    with st.sidebar:
        st.subheader("⚙️ Chat Controls")

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        st.divider()

        st.subheader("ℹ️ About")
        st.info("""
        This intelligent assistant uses:
        - **RAG Architecture**: Retrieval-Augmented Generation
        - **Vector Database**: ChromaDB for semantic search
        - **LLM**: Azure OpenAI GPT-4
        - **Memory**: Conversation context tracking
        - **Enhanced Retrieval**: MMR for diverse results
        """)

        st.divider()

        st.subheader("📊 Session Info")
        st.metric("Messages", len(st.session_state.chat_history))


if __name__ == "__main__":
    main()
