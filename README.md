# TDS Teaching Assistant (TDS-TA)

A comprehensive AI-powered teaching assistant system for the Tools in Data Science course, built with FastAPI and leveraging multiple LLM providers for intelligent course content assistance.

## 📋 Overview

TDS-TA is an intelligent teaching assistant that processes course forum data to provide contextual answers to student questions. The system scrapes forum content, processes it through an embedding pipeline, and serves responses via a FastAPI-based API.

##  Project Structure

```
tds-ta/
├── api/                     # FastAPI application
│   └── main.py             # Main API server with query endpoints
├── data_source/            # Data processing pipeline
│   ├── scrape.py          # Forum content scraper
│   ├── html_to_md.py      # HTML to text converter with image processing
│   ├── chunking.py        # Text chunking for embeddings
│   ├── embed.py           # Text embedding generation
│   └── embed2.py          # Alternative embedding implementation
├── deploy/                 # Vercel deployment files
│   ├── main.py            # Deployment-optimized API
│   ├── requirements.txt   # Minimal production dependencies
│   └── vercel.json        # Vercel configuration
├── main.py                # Simple entry point
├── analyze_chunks.py      # Comprehensive chunk analysis tool
├── prepare_deploy.sh      # Deployment preparation script
└── promptfoo.yaml         # Evaluation configuration
```

## 🚀 Features

- **Multi-modal Support**: Handles both text queries and image inputs
- **Smart Content Processing**: Converts HTML forum content to clean text with image descriptions
- **Intelligent Chunking**: Token-aware text segmentation for optimal embedding performance
- **Vector Search**: Semantic search using OpenAI embeddings
- **Multiple LLM Support**: Integration with OpenAI GPT models and Google Gemini
- **Rate Limiting**: Built-in API rate limiting for external services
- **Comprehensive Analysis**: Tools for evaluating chunking effectiveness
- **Production Ready**: Vercel deployment configuration included

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.13+
- **LLM Providers**: OpenAI GPT-4, Google Gemini
- **Embeddings**: OpenAI text-embedding-3-small
- **Data Processing**: BeautifulSoup, tiktoken, numpy
- **Deployment**: Vercel, UV package manager
- **Testing**: Promptfoo for evaluation

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd tds-ta
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ``

4. **Required API Keys**:
   - `OPENAI_API_KEY`: OpenAI API access
   - `GEMINI_API_KEY`: Google Gemini API access
## 🎯 Usage
### Running the API Server
```bash
cd api
python main.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `GET /` - Root endpoint with basic information
- `GET /health` - Health check endpoint
- `POST /api/` - Main query endpoint

#### Query Example

```bash
curl -X POST "https://tds-ta-tau.vercel.app/api/" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the TDS course about?",
    "image": "base64_encoded_image_optional"
  }'
```

#### Response Format

```json
{
  "answer": "Detailed answer to the question",
  "links": [
    {
      "url": "https://discourse.onlinedegree.iitm.ac.in/t/topic/123/1",
      "text": "Relevant forum post title"
    }
  ]
}
```

### Data Processing Pipeline

1. **Scrape Forum Data**:
   ```bash
   cd data_source
   python scrape.py
   ```

2. **Process HTML to Text**:
   ```bash
   python html_to_md.py
   ```

3. **Create Chunks**:
   ```bash
   python chunking.py
   ```

4. **Generate Embeddings**:
   ```bash
   python embed.py
   ```

5. **Analyze Chunk Quality**:
   ```bash
   python ../analyze_chunks.py
   ```

## 🔧 Configuration

### Chunking Parameters

- **MAX_TOKENS**: 500 (maximum tokens per chunk)
- **OVERLAP**: 30 (overlap between consecutive chunks)
- **Embedding Model**: text-embedding-3-small

### Rate Limiting

- **Gemini API**: 15 requests per 60 seconds
- Automatic backoff and retry mechanisms

## 📊 Analysis Tools

The project includes comprehensive analysis tools for evaluating chunking effectiveness:

```bash
python analyze_chunks.py
```

**Analysis Features**:
- Token distribution statistics
- Content preservation assessment
- Overlap analysis between chunks
- Content type classification
- Recommendations for optimization

## 🚀 Deployment

### Vercel Deployment

1. **Prepare deployment**:
   ```bash
   ./prepare_deploy.sh
   ```

2. **Deploy to Vercel**:
   - Upload the `deploy/` directory
   - Set environment variables in Vercel dashboard
   - Configure custom domains if needed

### Environment Variables for Production

```bash
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## 🧪 Testing & Evaluation

The project uses Promptfoo for systematic evaluation:

```bash
promptfoo eval -c promptfoo.yaml
```

**Evaluation Features**:
- Automated response quality assessment
- JSON schema validation
- Performance benchmarking
- Comparative analysis

## 📁 Key Files

- **`api/main.py`**: Main FastAPI application with endpoints
- **`data_source/scrape.py`**: Forum content scraper
- **`data_source/embed.py`**: Embedding generation pipeline
- **`analyze_chunks.py`**: Comprehensive chunk analysis tool
- **`prepare_deploy.sh`**: Automated deployment preparation
- **`promptfoo.yaml`**: Evaluation configuration

## 🔍 Data Processing Details

### Content Sources

The system processes data from the Tools in Data Science course forum, specifically:
- **Category**: `courses/tds-kb` (Category ID: 34)
- **Date Range**: January 1, 2025 - April 14, 2025
- **Content Types**: Q&A posts, assignments, discussions, code examples

### Processing Pipeline

1. **Scraping**: Extracts raw HTML from forum posts
2. **Cleaning**: Converts HTML to clean text, processes images with Gemini Vision
3. **Chunking**: Splits content into token-limited chunks with overlap
4. **Embedding**: Generates vector embeddings for semantic search
5. **Storage**: Compressed NPZ format for efficient retrieval

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and analysis tools
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.

## 🙋‍♂️ Support

For issues and questions:
- Check the analysis reports for data quality insights
- Review the comprehensive logging output
- Ensure all API keys are properly configured
- Verify rate limits are not exceeded

---

