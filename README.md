# Realestate Property Chatbot

An AI-powered chatbot that answers questions about Mahindra Lifespaces properties using Azure OpenAI, LangChain, and vector search.

## Overview

This chatbot provides intelligent answers about 13+ Mahindra Lifespaces properties across India. It uses RAG (Retrieval-Augmented Generation) to combine semantic search with AI to deliver accurate, context-aware responses.

## Features

- Natural language Q&A about properties
- Conversation memory for follow-up questions
- Semantic search using vector embeddings
- Well-formatted responses with hyperlinks
- Support for 13+ properties across Pune, Chennai, Bangalore, Mumbai, Gurgaon, and more

## Technology Stack

- **Frontend**: Streamlit
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **LLM**: Azure OpenAI GPT-4
- **Framework**: LangChain

## Prerequisites

- Python 3.8+
- Azure OpenAI API key
- 2GB+ RAM

## Installation

**1. Install Dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure Environment**

Create a `.env` file:

```env
AZURE_OPENAI_KEY=your_azure_openai_api_key
```

**3. Create Vector Database**

```bash
python dbmaker.py
```

## Usage

**Start the Application**

```bash
streamlit run app3.py
```

Open `http://localhost:8501` in your browser.

**Example Questions**

- "What properties are available in Pune?"
- "Tell me about Mahindra Citadel amenities"
- "What's the price range for 2BHK in Chennai?"
- "Compare Citadel and Happinest Tathawade"

## Project Structure

```
final realestate/
├── app3.py                      # Main Streamlit application
├── dbmaker.py                   # Vector database creation
├── mahendra_rechunked2.json     # Property data
├── requirements.txt             # Dependencies
├── .env                         # API keys
└── chroma_db/                   # Vector database (auto-generated)
```

## Supported Properties

**Pune**: Mahindra Citadel, Happinest Tathawade
**Chennai**: Aqualily, Happinest Avadi, Green Estates, Lakefront Estates
**Gurgaon**: Mahindra Aura
**Bangalore**: Mahindra NewHaven
**Mumbai/Alibaug**: Happinest Palghar, Happinest Boisar, Mahindra Meridian
**Origins**: Ahmedabad, Chennai

## Troubleshooting

**Database not found**: Run `python dbmaker.py` to create the vector database

**API Error**: Check your `AZURE_OPENAI_KEY` in the `.env` file

**First run is slow**: The embedding model (120MB) downloads on first run

**Out of memory**: Ensure 2GB+ RAM is available

