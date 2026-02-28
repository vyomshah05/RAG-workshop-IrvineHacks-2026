# UCI Tour Guide RAG Workshop Demo  
### Retrieval-Augmented Generation (RAG) — Irvine Hacks Workshop

Welcome! This repository accompanies the **“Build Your Own RAG System”** workshop at Irvine Hacks.

In this workshop, we build a **UC Irvine Tour Guide chatbot** powered by Retrieval-Augmented Generation (RAG). The goal is educational: to clearly demonstrate how modern AI applications combine:

- Your own documents  
- Semantic search (embeddings + vector databases)  
- A local Large Language Model (LLM)  

This is **not** a production application — it is a minimal, transparent implementation designed for learning and hackathon inspiration.

---

# What You’ll Learn

By running this demo, you’ll see how to:

1. Load and chunk documents
2. Convert text into embeddings
3. Store embeddings in a vector database (FAISS)
4. Retrieve relevant chunks for a question
5. Send retrieved context to an LLM
6. Generate grounded answers with citations

This is the core architecture behind many real-world AI tools.

---

# What is RAG?

**Retrieval-Augmented Generation (RAG)** combines:

- A **retriever** (search engine over your data)
- A **generator** (LLM that writes the answer)

Instead of asking an LLM directly:


LLM(question)


We do:


Retrieve relevant info → Give to LLM → Generate grounded answer


This:
- Reduces hallucinations  
- Grounds answers in your data  
- Makes AI useful for hackathon projects  

---

# Tech Stack (Free + Local)

This workshop prioritizes speed and zero cost.

| Component | Tool |
|------------|-------|
| LLM (generation) | Ollama (local model) |
| Embeddings | sentence-transformers |
| Vector Database | FAISS (CPU) |
| Language | Python |
| Interface | Terminal chatbot |

Everything runs locally.

---

# Setup Instructions

## 1. Install Ollama

Download and install Ollama from its official site.

After installing, pull a small model:

```
ollama pull llama3.2:3b
```

If that tag doesn’t exist on your system, use any small instruct model available in:

ollama list
## 2. Create a Virtual Environment

From the project directory:

```
python -m venv venv
```
```
source venv/bin/activate
```

On Windows:

```
venv\Scripts\activate
```

## 3. Install Dependencies

```
pip install -r requirements.txt
```

### Step 1: Ingest the Documents

This builds the vector database from the files in data/.

```
python ingest.py
```

This script:

- Loads text and PDFs from data/

- Splits them into chunks

- Converts chunks into embeddings

- Stores them in a FAISS index

You should see:

`Loaded X documents -> Y chunks`

Ingestion complete.

### Step 2: Run the Chatbot

```
python app.py
```

You’ll see:

`UCI Tour Guide RAG Bot (local + free)`

Now ask questions like:


- Where is the center of campus?

- Where do students study?

- Any tips for walking between classes?


To exit:

```
exit
```

### What Happens Behind the Scenes

When you ask a question:

1. The question is converted into an embedding

2. FAISS retrieves the most similar chunks

3. Retrieved context is added to a prompt

4. Ollama generates a grounded response

5. Sources are printed

7. If the answer is not in the context, say you don’t know. Try asking something unrelated (e.g., “Best boba near campus?”) to see hallucination prevention in action.

## Folder Structure

```
uci-rag-demo/
│
├── data/           # UCI tour guide documents
├── index/          # Generated FAISS index (after ingestion)
├── ingest.py       # Builds vector database
├── rag.py          # Retrieval + generation logic
├── app.py          # Chat interface
├── requirements.txt
└── README.md
```

# How to Extend This for Hackathons

After understanding this demo, you can:

Replace data/ with:

- PDFs

- Class notes

- API docs

- Product documentation

- Research papers

- Add a web UI (Streamlit or React frontend)

- Add chat history memory

- Add citation highlighting

- Deploy it as a web app

- RAG works with any text data.

# Workshop Philosophy

This repository intentionally avoids:

- Heavy frameworks (LangChain, LlamaIndex)

- Paid APIs

- Complex deployment

**The goal is clarity.**

You should be able to open each file and understand exactly:

- Where chunking happens

- Where embeddings are created

- Where retrieval occurs

- Where the LLM is called

If you understand this version, you understand RAG.

# Why This Makes a Great Hackathon Feature

Judges love:

- AI features

- Real usefulness

- Grounded answers

- Interactive demos

RAG delivers all four.

# Final Takeaway

**RAG is not magic.**

It’s:

- Search + LLM

- And once you understand that, you can build powerful AI systems with your own data.

**Happy hacking at Irvine Hacks! 🚀**
