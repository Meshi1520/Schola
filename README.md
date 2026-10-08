# Schola 📚

**Schola is an AI-powered RAG application that gives students instant, sourced answers and summaries from their own study materials.**

Upload your textbooks as PDFs, ask a question in plain English, and get a streamed answer along with the exact book and page it came from.

🔗 **Live demo:** _add your Streamlit link here_

![Schola screenshot](screenshot.png)

---

## Features

- **Chat with your books**: ask questions and get answers grounded in your own PDFs, not the open internet
- **Source citations**: every answer shows the book title, page number, and the passage it was drawn from
- **Streaming responses**: answers appear token by token, so there is no waiting on a blank screen
- **Add and remove books**: manage your library from the UI without touching code
- **Fast inference**: powered by Groq for low-latency responses

## How it works

```
PDF  →  split into chunks  →  embeddings  →  Chroma vector store
                                                     │
Your question  →  retrieve top matching chunks  ─────┘
                           │
                           ▼
        Prompt (question + retrieved context)  →  LLM  →  streamed answer + sources
```

1. **Ingest**: PDFs are loaded and split into overlapping chunks, with the book title and page number kept as metadata.
2. **Index**: each chunk is embedded and stored in a Chroma vector database.
3. **Retrieve**: your question is embedded and the most similar chunks are fetched.
4. **Generate**: the chunks are passed to the LLM as context, and the answer is streamed back with the sources shown below it.

## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| Vector store | ChromaDB |
| LLM | Groq (`gpt-oss-120b`) |
| Language | Python |

## Getting started

**1. Clone the repo**

```bash
git clone https://github.com/Meshi1520/Schola.git
cd Schola
```

**2. Create a virtual environment and install dependencies**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Add your API key**

Get a free key from the [Groq Console](https://console.groq.com), then create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

(On Streamlit Cloud, add it under **Settings → Secrets** instead.)

**4. Run the app**

```bash
streamlit run main.py
```

Open the local URL it prints, upload a PDF, and start asking questions.

## Usage tips

- Ask specific questions ("Explain backpropagation as it is described in chapter 4") for the best retrieval.
- Use the sources panel to verify an answer against the original page.
- Remove a book from the sidebar to stop it from being searched.

## Known limitations

- On Streamlit Cloud the disk is temporary, so uploaded books and the vector store reset when the app restarts.
- Scanned PDFs without a text layer are not supported, since there is no OCR step.
- Answer quality depends on retrieval, so very broad questions may miss relevant passages.

## Roadmap

- [ ] Persistent storage for uploaded books
- [ ] OCR support for scanned PDFs
- [ ] Conversation memory for follow-up questions
- [ ] Exportable summaries and study notes
- [ ] Retrieval evaluation to tune chunk size and top-k

## Author

Built by **Shivansh** ([@Meshi1520](https://github.com/Meshi1520)).


