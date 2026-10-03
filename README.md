# HR Policy Assistant using RAG

A Streamlit-based Retrieval-Augmented Generation (RAG) application that allows users to upload an HR Policy PDF and ask natural-language questions about the policy.

The assistant retrieves relevant portions of the uploaded policy using Sentence Transformers and FAISS, then sends only the retrieved policy context to Groq's `openai/gpt-oss-20b` model.

## Architecture

User uploads HR Policy PDF

↓

PyMuPDF extracts text page-by-page

↓

Policy text is divided into overlapping chunks

↓

Sentence Transformers creates embeddings

↓

FAISS stores and searches the embeddings

↓

User asks a question

↓

Question is converted to an embedding

↓

FAISS retrieves the most relevant policy chunks

↓

Retrieved chunks + question are sent to Groq

↓

GPT-OSS 20B generates a policy-grounded answer

## Technology Stack

- Streamlit
- FAISS
- Sentence Transformers
- PyMuPDF
- Groq API
- OpenAI GPT-OSS 20B

## Embedding Model

The project uses:

`sentence-transformers/all-MiniLM-L6-v2`

This is a lightweight Sentence Transformer suitable for semantic search and RAG applications.

## LLM

The application uses Groq with:

`openai/gpt-oss-20b`

## RAG Behaviour

The model is instructed to answer only from retrieved HR Policy excerpts.

If the retrieved policy does not contain enough information, the assistant responds:

> I could not find that in the uploaded HR policy.

The assistant also cites relevant PDF page numbers where possible.

## Groq API Key

Do not place your Groq API key inside `app.py`.

When deploying on Streamlit Community Cloud, add the API key through Streamlit's Secrets settings:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

## Deployment

This project can be deployed directly from GitHub to Streamlit Community Cloud.

The required files are:

```text
hr-policy-assistant/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

Set the Streamlit entrypoint to:

`app.py`

Python 3.11 is recommended.

## Important Limitation

This version extracts selectable text from PDFs.

Scanned or image-only HR Policy PDFs require OCR and are not supported by the current version.

## Privacy

The uploaded PDF and FAISS vector index are created in the running Streamlit session. They are not committed to the GitHub repository by this application.

Relevant retrieved policy excerpts are sent to the Groq API when a user asks a question.

## Disclaimer

This application is intended to help users search and understand HR policy documents. It should not replace official HR, legal, or organizational advice where authoritative interpretation is required.
