# Mahindra Lifespaces Property Chatbot

An intelligent AI-powered chatbot that provides comprehensive, context-aware answers about 13+ Mahindra Lifespaces properties across India. Built with Azure OpenAI and LangChain, it uses advanced RAG (Retrieval-Augmented Generation) with MMR (Maximal Marginal Relevance) retrieval for diverse yet relevant results. Features include conversation memory, smart deduplication, automatic hyperlink generation, and clean markdown formatting with ChromaDB vector search.

## Features

- **Natural Language Q&A**: Ask questions about properties in plain English
- **Conversation Memory**: Maintains context across multiple questions (last 6 messages)
- **Enhanced Retrieval**: MMR-based search for diverse yet relevant results
- **Smart Deduplication**: Removes redundant information from search results
- **Hyperlinked Responses**: Automatically links property names to official pages
- **Clean Formatting**: Well-structured markdown responses with tables and bullet points
- **13+ Properties**: Coverage across Pune, Chennai, Bangalore, Mumbai, Gurgaon, Ahmedabad, and Alibaug

## Technology Stack

- **Frontend**: Streamlit with chat interface
- **Vector Database**: ChromaDB with persistence
- **Embeddings**: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
- **LLM**: Azure OpenAI GPT-4.1-mini
- **Framework**: LangChain
- **Retrieval Strategy**: MMR (λ=0.7, k=15, fetch_k=30)

## Prerequisites

- Python 3.8+
- Azure OpenAI API key and endpoint access
- 2GB+ RAM (for embedding model)

## Installation

**1. Install Dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure Environment**

Create a `.env` file in the project root:

```env
AZURE_OPENAI_KEY=your_azure_openai_api_key
```

**Note**: The Azure endpoint is configured to `https://inadev-saraswati-openai.openai.azure.com/` by default. Update in `app.py` if using a different endpoint.

**3. Create Vector Database**

```bash
python dbmaker.py
```

This will process `mahendra_rechunked2.json` and create the `chroma_db/` directory.

## Usage

**Start the Application**

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**Example Questions**

- "What properties are available in Pune?"
- "Tell me about Mahindra Citadel amenities"
- "What's the price range for 2BHK in Chennai?"
- "Compare Citadel and Happinest Tathawade"

## Project Structure

```
Real-estate-chatbot/
├── app.py                       # Main Streamlit application
├── dbmaker.py                   # Vector database creation script
├── mahendra_rechunked2.json     # Property data (source)
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (API keys)
└── chroma_db/                   # ChromaDB vector store (auto-generated)
```

## Architecture

### Enhanced RAG Pipeline

1. **Retrieval Phase**:
   - MMR (Maximal Marginal Relevance) search for diverse results
   - Fetches 30 candidates, returns top 15 (λ=0.7 for relevance-diversity balance)
   - Automatic fallback to similarity search if MMR fails
   - Smart deduplication based on content similarity

2. **Prompt Engineering**:
   - Structured system prompt with authoritative URL dictionary
   - Conversation history context (last 6 messages)
   - Retrieved chunks with metadata and source tracking
   - Strict guidelines for hyperlink generation and formatting

3. **Response Generation**:
   - Azure OpenAI GPT-4.1-mini (temperature=0.1, max_tokens=2000)
   - Markdown formatting with proper hyperlinks
   - Error handling with graceful degradation

### Key Components

- **`safe_load_embeddings()`**: Cached HuggingFace embedding model loader (app.py:26)
- **`safe_load_db()`**: Cached ChromaDB initialization (app.py:37)
- **`retrieve_chunks_enhanced()`**: MMR-based retrieval with fallback (app.py:83)
- **`deduplicate_and_rank_chunks()`**: Content deduplication (app.py:110)
- **`build_enhanced_prompt()`**: Advanced prompt construction with URL mapping (app.py:130)
- **`rag_answer_enhanced()`**: Complete RAG pipeline (app.py:243)

## Supported Properties

The chatbot has comprehensive information about the following properties with official URL linking:

| Location | Properties |
|----------|-----------|
| **Pune** | Mahindra Citadel, Mahindra Happinest Tathawade |
| **Chennai** | Mahindra Aqualily, Mahindra Happinest Avadi, Green Estates by Mahindra, Lakefront Estates |
| **Gurgaon** | Mahindra Aura |
| **Bangalore** | Mahindra NewHaven |
| **Mumbai** | Mahindra Happinest Palghar, Mahindra Happinest Boisar |
| **Alibaug** | Mahindra Meridian |
| **Ahmedabad** | Origins by Mahindra – Ahmedabad |
| **Chennai (Plots)** | Origins by Mahindra – Chennai |

All property mentions in responses are automatically hyperlinked to their official Mahindra Lifespaces pages.

## Configuration

### Environment Variables

```env
AZURE_OPENAI_KEY=your_api_key_here
```

### Azure OpenAI Settings (in app.py)

```python
AZURE_ENDPOINT = "https://inadev-saraswati-openai.openai.azure.com/"
AZURE_VERSION = "2024-12-01-preview"
AZURE_DEPLOYMENT = "gpt-4.1-mini"
```

### Retrieval Parameters

- **k**: 15 (number of chunks to retrieve)
- **fetch_k**: 30 (MMR candidate pool size)
- **lambda_mult**: 0.7 (relevance vs diversity balance)
- **temperature**: 0.1 (LLM temperature for consistent responses)
- **max_tokens**: 2000 (response length limit)

## Troubleshooting

### Database Issues

**Error**: `Chroma DB load failed`
**Solution**: Run `python dbmaker.py` to create the vector database. Ensure `mahendra_rechunked2.json` exists.

### API Issues

**Error**: `401 Unauthorized` or API key errors
**Solution**:
1. Verify `AZURE_OPENAI_KEY` in `.env` file
2. Check Azure endpoint URL matches your deployment
3. Confirm deployment name is correct (`gpt-4.1-mini`)

### Performance Issues

**Issue**: First run is slow (1-2 minutes)
**Cause**: HuggingFace embedding model (sentence-transformers/all-MiniLM-L6-v2, ~120MB) downloads on first run
**Solution**: Wait for initial download; subsequent runs use cached model

**Issue**: Out of memory errors
**Solution**: Ensure minimum 2GB RAM available. Close other applications if needed.

### Retrieval Issues

**Issue**: "No relevant information found"
**Solution**:
1. Rephrase query to be more specific
2. Check if `chroma_db/` directory exists and is populated
3. Verify vector database was created successfully with `dbmaker.py`

### Streamlit Issues

**Error**: Module not found
**Solution**: Run `pip install -r requirements.txt` to install all dependencies

**Error**: Port already in use
**Solution**: Use `streamlit run app.py --server.port 8502` to use a different port

## Advanced Features

### Session Controls

- **Clear Chat History**: Button in sidebar to reset conversation context
- **Session Metrics**: Display message count in sidebar
- **Persistent Cache**: `@st.cache_resource` decorators for model/DB loading

### Response Quality Features

1. **Smart URL Linking**: Case-insensitive partial matching (e.g., "Citadel" → "Mahindra Citadel")
2. **Source Attribution**: Each retrieved chunk includes metadata source
3. **Context-Aware**: Maintains last 6 messages for coherent multi-turn conversations
4. **Structured Output**: Automatic markdown formatting with headings, bullets, and tables

## Development

### Modifying the Knowledge Base

1. Edit `mahendra_rechunked2.json` with updated property information
2. Run `python dbmaker.py` to rebuild vector database
3. Restart the Streamlit app

### Adding New Properties

1. Add property data to `mahendra_rechunked2.json`
2. Add official URL to `VALID_URLS_DICT` in `app.py` (line 63)
3. Rebuild database: `python dbmaker.py`
4. Restart app

### Customizing Retrieval

Adjust parameters in `retrieve_chunks_enhanced()` (app.py:83):
- Increase `k` for more context (may reduce precision)
- Adjust `lambda_mult` (0.0 = diversity, 1.0 = relevance)
- Modify `fetch_k` for larger candidate pools

## Dependencies

Key packages (see `requirements.txt` for full list):
- `streamlit` - Web UI framework
- `langchain-community` - Vector store integration
- `langchain-openai` - Azure OpenAI integration
- `langchain-huggingface` - Embedding models
- `chromadb` - Vector database
- `python-dotenv` - Environment variable management

## License

This project is for Mahindra Lifespaces property information purposes.

