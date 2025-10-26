## 📚 Chat with Multiple PDFs — Streamlit RAG App

### 💡 Overview

A **Streamlit-based Retrieval-Augmented Generation (RAG) application** that allows users to **chat with multiple PDF documents** and get **context-aware answers** directly from the content of uploaded files.

This project integrates **LangChain**, **HuggingFace embeddings**, **FAISS vector storage**, and **LLMs (via OpenRouter API)** to deliver an interactive, memory-enabled chatbot experience.

---

### 🔍 Key Highlights

* 🧾 **Multi-PDF Chat Interface:**
  Upload and query multiple PDFs simultaneously with seamless context switching.

* 🧠 **Complete RAG Pipeline:**
  Implements every step — from document loading and text chunking to embedding, retrieval, and response generation — using **LangChain**.

* ⚙️ **Efficient Retrieval:**
  Uses **HuggingFace embeddings** (`all-MiniLM-L6-v2`) and **FAISS** for fast and scalable semantic search over document chunks.

* 💬 **Conversational Memory:**
  Maintains context using **ConversationBufferMemory**, enabling **multi-turn dialogue** that understands prior messages.

* 🤖 **LLM Integration via OpenRouter:**
  Supports top open models such as:

  * `meta-llama/llama-3.3-8b-instruct:free`
  * `meta-llama/llama-3.3-70b-instruct:free`
  * `qwen/qwen3-14b:free`

* 🧩 **Smart Text Handling:**
  Uses **RecursiveCharacterTextSplitter** to handle large documents efficiently by splitting text into overlapping, meaningful chunks.

* 🎨 **User-Friendly Streamlit UI:**
  Clean and intuitive interface for uploading PDFs, selecting models, and chatting with contextual responses.

 
---

## 🧰 Tech Stack

| Component          | Description                                 |
| ------------------ | ------------------------------------------- |
| **Frontend**       | Streamlit                                   |
| **LLM Framework**  | LangChain                                   |
| **Embeddings**     | HuggingFace (`all-MiniLM-L6-v2`)            |
| **Vector Store**   | FAISS                                       |
| **Model Provider** | OpenRouter API (supports LLaMA, Qwen, etc.) |
| **PDF Processing** | PyPDF2                                      |

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/chat-with-multiple-pdfs.git
cd chat-with-multiple-pdfs
```

### 2️⃣ Create and Activate a Virtual Environment

```bash
# Windows
python -m venv chat_pdf
chat_pdf\Scripts\activate

# macOS/Linux
python3 -m venv chat_pdf
source chat_pdf/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Setup

Create a `.env` file in the project root and add your **OpenRouter API key**:

```
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_API_BASE=https://openrouter.ai/api/v1
```

You can get your API key from [https://openrouter.ai/](https://openrouter.ai/).

---

## 🧠 Local Model Setup (Optional)

If you prefer using local embeddings:

1. Download the model [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
2. Place it in your local model directory, for example:

   ```
   C:\Users\<your_name>\Projects\model\all-MiniLM-L6-v2
   ```
3. The path is already referenced in `get_vectorstore()` in the code.

---

## ▶️ Run the App

Once everything is set up, run:

```bash
streamlit run app.py
```

Then open your browser and go to **[http://localhost:8501/](http://localhost:8501/)**.

---

## 💬 Usage Guide

1. Upload one or more PDF files from the sidebar.
2. Select a language model (e.g., `meta-llama/llama-3.3-8b-instruct:free`).
3. Click on **🚀 Process** to embed and index the documents.
4. Ask any question in the text box (e.g., *“Summarize Chapter 2”* or *“What are the key findings?”*).
5. Enjoy a contextual, multi-turn conversation powered by RAG!

---

## 🧩 Project Structure

```
📦 chat-with-multiple-pdfs
│
├── app.py                      # Main Streamlit app
├── htmlTemplates.py            # Custom chat UI templates (bot/user)
├── requirements.txt            # Python dependencies
├── .env                        # API keys and environment config
└── README.md                   # Project documentation
```


## 🛠️ Key Functions

| Function                   | Purpose                                           |
| -------------------------- | ------------------------------------------------- |
| `get_pdf_text()`           | Extracts raw text from uploaded PDF files.        |
| `get_text_chunks()`        | Splits long text into overlapping chunks.         |
| `get_vectorstore()`        | Embeds and stores text chunks in FAISS.           |
| `get_conversation_chain()` | Builds the LangChain conversational RAG pipeline. |
| `handle_userinput()`       | Handles user queries and displays chat messages.  |

---

## 🌐 Model Options via OpenRouter

| Model                                    | Description                         |
| ---------------------------------------- | ----------------------------------- |
| `meta-llama/llama-3.3-70b-instruct:free` | Large instruction-tuned LLaMA model |
| `meta-llama/llama-3.3-8b-instruct:free`  | Faster, smaller version of LLaMA    |
| `qwen/qwen3-14b:free`                    | High-quality general-purpose model  |

---

## 📸 App Preview

*(You can add a screenshot or GIF here later)*

```
📄 Upload PDFs → 🧠 Process → 💬 Chat → 🔄 Continue Contextual Conversation
```


## 🪪 License

This project is open-source and available under the **MIT License**.
