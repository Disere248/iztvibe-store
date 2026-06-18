# IZTVibe Store — AI-Driven E-Commerce MVP

A conceptual e-commerce platform that replaces traditional hardcoded filters with an adaptive, natural language assistant. The system filters items locally using mathematical semantic analysis and processes recommendations through high-performance LLM infrastructure.

---

## Technological Architecture

The repository is built following the Separation of Concerns (SoC) principle, splitting interface, data, and processing logic into independent modules:

* main.py — The presentation layer built on Streamlit, managing states, onboarding UI dialogues, and reactive chat elements.
* ai_logic.py — The core analytical engine handling tokenization, vector mechanics, and LLM orchestration.
* products.py — The isolated database layer storing structured product contexts and descriptions.

---

## Hybrid Semantic Pipeline (RAG)

To optimize tokens and minimize API latency, the system utilizes a Retrieval-Augmented Generation (RAG) pipeline before communicating with the cloud model:

1. Local Tokenization & Processing: The user's input string is cleaned, stripped of punctuation, and tokenized into individual semantic units.
2. Vector Similarity Evaluation: The system computes the Cosine Similarity between the user query vector (A) and each product's metadata vector (B) using the formula:
3. Context Construction: The top N products exceeding the relevance threshold are selected. Their deep characteristics (materials, exact prices, specific descriptions) are packed into an injection context along with a global list of available titles to guarantee model alignment.
4. Inference via Groq API: The structured context and the original query are passed to the ultra-fast Groq API running the llama-3.1-8b-instant model, which acts as a professional stylist.

---

## Tech Stack & Dependencies

* Language: Python 3.11+
* UI Framework: Streamlit (Reactive Web UI)
* LLM Provider: Groq Cloud API
* Inference Model: Llama 3.1 8B Instant (llama-3.1-8b-instant)
* Mathematics: Pure Python implementation of vector dot-products and algebraic lengths (zero external heavy dependencies like NumPy or SciPy for maximum lightweight deployment).

---

## Quick Start & Installation

### 1. Clone the repository
```bash
git clone [https://github.com/disere248/iztvibe-store.git](https://github.com/disere248/iztvibe-store.git)
cd iztvibe-store

```
### 2. Configure environment & dependencies

Create a virtual environment and install the required official SDK packages:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install streamlit groq

```
### 3. Launch the Application
```bash
streamlit run main.py

```
### 4. API Key Access
To fully experience the semantic assistant, get your API Key from the official Groq Cloud Console and enter it into the secure password field in the application sidebar.
```