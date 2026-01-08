# Synthetic QA Dataset Generator

A Python application for generating synthetic Question-Answer (QA) datasets from PDF documents using LLMs. The tool intelligently extracts content, identifies semantically similar chunks, and generates high-quality QA pairs through an evolutionary refinement process.

## Features

- **PDF Processing**: Extracts text from PDFs using Docling with OCR support
- **Smart Chunking**: Word-aware splitting that respects legal document structure (headers, sections, articles)
- **Semantic Similarity**: Uses embeddings to find relevant context chunks
- **Question Evolution**: Refines questions through multiple steps using various strategies
- **Flexible LLM Integration**: Supports OpenRouter, local models, or any OpenAI-compatible API

## How It Works

1. **Document Processing**: Converts PDF to markdown and splits into semantic chunks
2. **Embedding**: Embeds chunks using your chosen embedding model
3. **Context Selection**: Randomly selects a chunk and finds similar chunks (>80% cosine similarity)
4. **Question Generation**: Creates multiple questions based on the selected contexts
5. **Question Evolution**: Refines questions through N steps using random refinement strategies
6. **Answer Generation**: Generates comprehensive answers using the contexts

## Installation

```bash
pip install -r requirements.txt

## or use uv with 2 approaches below

## install global
uv pip install -r requirements.txt

## install to current venv
uv add -r requirements.txt
```

The application will automatically download the required spaCy model (`en_core_web_sm`) on first run if not present.

## Usage

### Basic Command

```bash
## ensure you are on root of the project or adjust accordingly
python -m src.main \
  --file document.pdf \
  --generator_model "anthropic/claude-3.5-sonnet" \
  --embedder_model "text-embedding-3-small" \
  --epoch 5
```

### Full Options

```bash
python app.py \
  --file <path-to-pdf> \
  --generator_model <model-name> \
  --base_url_generator <api-endpoint> \
  --embedder_model <embedding-model> \
  --base_url_embedder <embedding-endpoint> \
  --epoch <number-of-iterations> \
  --question_evolve_steps <refinement-steps> \
  --api_key_generator <your-api-key> \
  --api_key_embedding <your-embedding-key>
```

### Parameters

| Parameter                 | Description                                                    | Default                        |
| ------------------------- | -------------------------------------------------------------- | ------------------------------ |
| `--file`                  | Path to the PDF document                                       | Required                       |
| `--generator_model`       | LLM model for generation (e.g., `anthropic/claude-3.5-sonnet`) | Required                       |
| `--base_url_generator`    | API endpoint for generator model                               | `https://openrouter.ai/api/v1` |
| `--embedder_model`        | Embedding model name                                           | Required                       |
| `--base_url_embedder`     | API endpoint for embedder                                      | `http://localhost:8000/v1`     |
| `--epoch`                 | Number of generation iterations                                | `3`                            |
| `--question_evolve_steps` | Number of refinement steps per question                        | `2`                            |
| `--api_key_generator`     | API key for generator (prompted if needed)                     | None                           |
| `--api_key_embedding`     | API key for embedder (prompted if needed)                      | None                           |

## Required Template Files

The application requires Jinja2 templates in the working directory:

### 1. `question-prompt.jinja`
Used for initial question generation. **Must include these variables:**
- `contexts` - List of context chunks
- `num_question_per_context` - Number of questions to generate

### 2. `refine_question_*.jinja`
Templates for question evolution (e.g., `refine_question_complexity.jinja`, `refine_question_reasoning.jinja`). **Must include:**
- `contexts` - List of context chunks
- `original_input` - The question to refine

Multiple refinement templates can exist; the app randomly selects one per evolution step.

### 3. `expected-output-prompt.jinja`
Used for answer generation. **Must include:**
- `original_input` - The question
- `contexts` - List of context chunks

## Example Templates

### question-prompt.jinja
```jinja
Generate {{ num_question_per_context }} questions based on these contexts:

{% for context in contexts %}
Context {{ loop.index }}:
{{ context }}

{% endfor %}

Return a JSON array of questions.
```

### refine_question_complexity.jinja
```jinja
Original question: {{ original_input }}

Contexts:
{% for context in contexts %}
{{ context }}
{% endfor %}

Make this question more complex and detailed.
```

### expected-output-prompt.jinja
```jinja
Question: {{ original_input }}

Answer this question using these contexts:
{% for context in contexts %}
{{ context }}
{% endfor %}
```

## Output

The application generates a JSON file named `<uuid>_<date>.json` with the following structure:

```json
[
  {
    "question": "What are the key provisions of Article 5?",
    "answer": "Article 5 establishes..."
  },
  {
    "question": "How does Section 3 relate to...",
    "answer": "Section 3 provides..."
  }
]
```

## Document Chunking

The `WordAwareLegalSplitter` intelligently handles:
- Legal document structures (PASAL, BAB, BAGIAN, etc.)
- Headers and section titles
- Sentence boundaries using spaCy
- Configurable chunk sizes (default: 150 words)
- Chunk overlap (default: 30 words)
- Automatic header-content merging to prevent orphaned headers

## API Key Handling

- **Localhost URLs**: No API key required (uses `'dummykey'`)
- **Remote APIs**: Will prompt for API key if not provided via command line
- Keys can be provided via `--api_key_generator` and `--api_key_embedding` options

## GPU Support

The PDF processor supports GPU acceleration for faster OCR:
- Automatically uses CUDA if available
- Configurable page batch size for memory management

## Tips

1. **Start Small**: Test with `--epoch 1` first to validate templates and setup
2. **Template Strategy**: Create multiple `refine_question_*` templates for diverse question evolution
3. **Context Quality**: The chunking strategy is optimized for legal documents but works for general text
4. **Model Selection**: Use stronger models (GPT-4, Claude 3.5) for better question quality
5. **Embeddings**: Local embedding servers (vLLM, Ollama) can reduce costs significantly

## Troubleshooting

**JSON Decode Error**: Check your template outputs - ensure the LLM returns valid JSON for questions

**Memory Issues**: Reduce `page_batch_size` in `DoclingPdfReader` initialization or process smaller documents

**No Refinement Templates Found**: Ensure files starting with `refine_question_` exist in the working directory

## License

MIT License - feel free to use and modify for your needs.