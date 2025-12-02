# Health Status Aggregator

An AI-driven system that analyzes and aggregates health status data from Power BI reports. It extracts key indicators from free-text descriptions using an LLM, compares recommendations with predefined statuses, and triggers RAG-based reanalysis when confidence is low.

## Features

- **LLM-Powered Analysis**: Extracts health indicators from free-text descriptions using GPT-4 (or mock service for development)
- **RAG Enhancement**: Retrieves similar historical cases using FAISS vector search when confidence is low
- **Human-in-the-Loop**: Flags ambiguous cases for human review with full context
- **Batch Processing**: Processes up to 1000 records with timing metrics
- **Clean Architecture**: 4-layer architecture (Domain → Application → Infrastructure → Presentation)

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key (optional - mock LLM available for development)

### Installation

```bash
# Clone the repository
cd health_status_aggregator

# Install dependencies
pip install -r requirements.txt

# Generate synthetic test data
python generate_data.py

# Start the server
uvicorn app.main:app --reload
```

### Access the API

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc
- **Health Check**: http://127.0.0.1:8000/

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/analyze/` | POST | Analyze a single health record |
| `/analyze/batch/` | POST | Batch analysis (file path or records array) |
| `/feedback/` | POST | Submit human feedback for a record |
| `/reviews/pending/` | GET | Get all pending human review requests |
| `/reviews/{record_id}/` | GET | Get detailed review information |
| `/reports/generate/` | POST | Generate comprehensive report with metrics |
| `/data/validate/` | POST | Validate a dataset file |
| `/aggregations/by-account/` | GET | Get account-level health aggregations |

## Usage Examples

### Analyze a Single Record

```bash
curl -X POST "http://127.0.0.1:8000/analyze/" \
  -H "Content-Type: application/json" \
  -d '{
    "record_id": 1,
    "account_id": "ACC001",
    "project_id": "PRJ001",
    "predefined_health": "Good",
    "free_text_details": "Project on schedule and under budget. Team morale is high."
  }'
```

**Response:**
```json
{
  "result": {
    "record_id": 1,
    "indicators": ["[+] 'on schedule' (w=3.0)", "[+] 'under budget' (w=3.0)"],
    "recommended_health": "Good",
    "confidence": 0.92,
    "mismatched": false,
    "reasoning": "Weighted scores: Good=8.0, Warning=0.0, Critical=0.0",
    "processing_time_ms": 52.3,
    "requires_rag": false,
    "requires_human_review": false
  },
  "similar_cases": null,
  "rag_context": null,
  "review_created": false
}
```

### Batch Analysis

```bash
curl -X POST "http://127.0.0.1:8000/analyze/batch/" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "app/data/health_dataset.json"}'
```

### Submit Human Feedback

```bash
curl -X POST "http://127.0.0.1:8000/feedback/" \
  -H "Content-Type: application/json" \
  -d '{
    "record_id": 123,
    "human_decision": "Warning",
    "reviewer_notes": "Budget concerns noted but not critical yet"
  }'
```

## Testing

### Run Unit Tests

```bash
pytest tests/test_unit.py -v
```

**Test Coverage:**
- MockLLMService classification logic (6 tests)
- Edge cases: empty text, case insensitivity, ambiguity (4 tests)
- RAG similarity conversion math (5 tests)
- Use case workflow (2 tests)
- End-to-end classification accuracy (1 test)

### Run API Tests

```bash
python test_api.py
```

### Run Benchmark

```bash
python benchmark.py
```

**Benchmark Criteria:**
| Metric | Target | Typical Result |
|--------|--------|----------------|
| Classification Accuracy (F1) | ≥ 80% | ~83% |
| Processing Throughput | ≤ 30s/record | ~0.005s/record |
| RAG Response Time | ≤ 5s/query | ~12ms/query |
| Edge Case Handling | Flagged | ✓ |

## Project Structure

```
health_status_aggregator/
├── app/
│   ├── domain/                 # Business entities & interfaces
│   │   ├── entities/           # Record, AnalysisResult, Feedback
│   │   └── interfaces/         # ILLMService, IRAGService, etc.
│   ├── application/            # Use cases
│   │   └── use_cases/          # AnalyzeRecord, BatchAnalyze, etc.
│   ├── infrastructure/         # External implementations
│   │   ├── services/           # LLM, RAG, Reporting services
│   │   └── persistence/        # In-memory repositories
│   ├── presentation/           # API layer
│   │   ├── api/                # FastAPI routes
│   │   └── schemas/            # Pydantic DTOs
│   ├── container.py            # Dependency injection
│   └── main.py                 # Application bootstrap
├── tests/
│   └── test_unit.py            # Unit tests (21 tests)
├── generate_data.py            # Synthetic data generator
├── benchmark.py                # Performance benchmark
├── test_api.py                 # API integration tests
├── ARCHITECTURE.md             # Architecture documentation & ADRs
└── requirements.txt
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key for GPT-4 |
| `USE_MOCK_LLM` | `true` | Use mock LLM for development |

### Key Settings

- **Confidence Threshold**: 85% - Below this, RAG is triggered
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **RAG Top-K**: 5 similar cases retrieved

## Architecture

The system follows **Clean Architecture** principles:

```
┌─────────────────────────────────────────────────────────┐
│                   Presentation Layer                     │
│                  (FastAPI REST API)                      │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                   Application Layer                      │
│                    (Use Cases)                           │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                    Domain Layer                          │
│              (Entities & Interfaces)                     │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                 Infrastructure Layer                     │
│           (LLM, RAG, Persistence Services)               │
└─────────────────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation including:
- System diagrams
- Process flows
- Architectural Decision Records (ADRs)

## How It Works

### Analysis Flow

1. **Input**: Receive health record with `predefined_health` and `free_text_details`
2. **LLM Analysis**: Extract indicators and classify health status
3. **Confidence Check**: If confidence < 85% or mismatched → trigger RAG
4. **RAG Enhancement**: Retrieve 5 similar historical cases
5. **Reanalysis**: LLM reanalyzes with similar case context
6. **Human Review**: If still ambiguous, create review request
7. **Output**: Return classification with confidence, indicators, and reasoning

### Classification Logic (MockLLMService)

The mock LLM uses weighted keyword scoring:

| Category | Example Keywords | Weights |
|----------|------------------|---------|
| **Good** | "on schedule", "under budget" | 3.0 |
| **Warning** | "slight delay", "resource constraints" | 2.0-2.5 |
| **Critical** | "budget overrun", "at risk of cancellation" | 3.5-4.0 |

**Priority**: Critical > Warning > Good (risk-averse)

## Data Generation

Generate 1000 synthetic records with realistic variability:

```bash
python generate_data.py
```

**Distribution:**
- 65% Clear cases (unambiguous indicators)
- 15% Ambiguous (mixed signals)
- 5% Typos (minor spelling errors)
- 3% Incomplete (minimal text)
- 7% Complex (multiple issues)
- 5% Intentional mismatch (predefined ≠ text sentiment)

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Submit a pull request
