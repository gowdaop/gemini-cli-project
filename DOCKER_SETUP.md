# Docker Setup Instructions

## Prerequisites
- Docker Desktop installed
- Git installed
- A running Milvus instance
- Google Cloud Platform account with enabled APIs:
  - Vertex AI API
  - Document AI API
  - Discovery Engine API (for enhanced RAG web search)

## Quick Start
1. Clone the repository
2. Ensure your Milvus instance is running and accessible from your host machine.
3. Run setup script: `bash config/scripts/setup.sh`
4. Update `.env` with your values. Ensure `MILVUS_HOST` and `MILVUS_PORT` are correctly set to point to your Milvus instance. If running Milvus on your local machine, `MILVUS_HOST` should be `host.docker.internal` inside the `.env` file when running the application via docker-compose.
5. Place Google Cloud credentials in `./config/credentials/`
6. Configure enhanced RAG settings:
   - Set `GOOGLE_API_KEY` for Vertex AI and Gemini API access
   - Set `VERTEX_SEARCH_ENGINE_ID` for web search functionality
   - Set `VERTEX_SEARCH_PROJECT_ID` to your GCP project ID
   - Ensure `VERTEX_SEARCH_AVAILABLE=true` and `ASYNC_HTTP_AVAILABLE=true`
7. Access application at `http://localhost:8000`

## Daily Development
- Start: `make up` or `docker-compose up -d`
- Stop: `make down` or `docker-compose down`
- Logs: `make logs` or `docker-compose logs -f`
- Shell: `make shell` or `docker-compose exec legal-simplifier bash`

## Enhanced RAG Features

The system now includes enhanced RAG capabilities:

### Web Search Integration
- **GCP Vertex AI Search**: Primary web search using Google's Discovery Engine
- **DuckDuckGo API**: Fallback web search for additional context
- **Dual Search Strategy**: Combines both search results for comprehensive answers

### Chat Enhancements
- **Context-Aware Responses**: Answers based on uploaded document context
- **Indian Law Focus**: Prioritizes Indian legal framework and precedents
- **Smart Fallbacks**: Uses RAG → Web Search → LLM Knowledge hierarchy

### Summary Improvements
- **Extended Summaries**: Up to 500 words with detailed analysis
- **Recommendations**: Practical legal advice and next steps
- **Structured Output**: Clear separation of summary and recommendations

## Troubleshooting
- If containers fail: `docker-compose down && docker-compose up --build`
- Clean restart: `make clean && make build && make up`
- Check RAG service status: `docker-compose logs legal-simplifier | grep -i "rag\|search"`
- Verify API keys: Ensure all Google Cloud APIs are enabled and credentials are valid