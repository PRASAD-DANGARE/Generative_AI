# AI Foundation & Core Concepts
### From Backend Engineer to Agentic & GenAI Systems Engineer — First Principles Notes

> **How to use this document:** This is your conceptual map of the entire AI landscape — from "what is AI" down to the specific tools an AI Engineer touches daily. Read top to bottom once to build the mental hierarchy, then use it as quick revision before interviews.

---

## Table of Contents

1. [The AI Hierarchy](#1-the-ai-hierarchy)
2. [What is AI?](#2-what-is-ai)
3. [Rule-Based AI](#3-rule-based-ai)
4. [Machine Learning (ML)](#4-machine-learning)
5. [Deep Learning (DL)](#5-deep-learning)
6. [Neural Network Architectures: ANN, CNN, RNN, Transformers](#6-neural-network-architectures)
7. [Discriminative vs Generative Models](#7-discriminative-vs-generative)
8. [Generative AI (GenAI)](#8-generative-ai)
9. [Natural Language Processing (NLP)](#9-nlp)
10. [Computer Vision](#10-computer-vision)
11. [Large Language Models (LLM)](#11-large-language-models)
12. [Why is it called "Large"?](#12-why-large)
13. [Why is it called a "Language Model"?](#13-why-language-model)
14. [How an LLM Generates a Response — The Flow](#14-how-llm-generates-response)
15. [Foundation Models vs Fine-Tuned Models](#15-foundation-vs-finetuned)
16. [Major LLM Providers and Models](#16-major-llm-providers)
17. [What an LLM Cannot Do Alone](#17-what-llm-cannot-do)
18. [Embeddings](#18-embeddings)
19. [Vector Databases](#19-vector-databases)
20. [RAG (Retrieval-Augmented Generation)](#20-rag)
21. [AI Agents](#21-ai-agents)
22. [Agentic AI](#22-agentic-ai)
23. [Multi-Modal AI](#23-multi-modal-ai)
24. [Types of AI Models by Modality](#24-types-of-models-by-modality)
25. [Prompt Engineering](#25-prompt-engineering)
26. [Fine-Tuning](#26-fine-tuning)
27. [MLOps](#27-mlops)
28. [LLMOps (the GenAI-specific layer)](#28-llmops)
29. [LangChain, LangGraph, LlamaIndex](#29-langchain-langgraph-llamaindex)
30. [Model Context Protocol (MCP)](#30-mcp)
31. [Hallucinations](#31-hallucinations)
32. [Context Window](#32-context-window)
33. [Tokens & Tokenization (quick primer)](#33-tokens-quick-primer)
34. [Open Source vs Closed Source Models](#34-open-vs-closed-source)
35. [Traditional ML vs GenAI — Side by Side](#35-traditional-ml-vs-genai)
36. [The AI Engineer's Toolkit — Complete Map](#36-ai-engineers-toolkit)
37. [The AI Engineer's Job — What It Actually Is](#37-ai-engineers-job)

---

## 1. The AI Hierarchy

Every term in this field is a subset of a broader term. Understanding this hierarchy is the single most important mental model — almost every interview question maps back to "where does X sit in this tree?"

```
Artificial Intelligence (AI)
│
├── Rule-Based AI
│
├── Machine Learning (ML) — subset of AI
│   │
│   ├── Traditional ML
│   │
│   └── Deep Learning (DL) — subset of ML
│       │
│       ├── NLP (Natural Language Processing)
│       ├── Computer Vision
│       │
│       └── Generative AI (GenAI) — subset of DL
│           │
│           ├── LLM (Large Language Models)
│           ├── RAG (Retrieval-Augmented Generation)
│           ├── AI Agents
│           └── Multi-Modal AI
```

**Why this hierarchy matters:** Every "subset of" relationship means the inner concept inherits constraints and capabilities from the outer one, plus adds something specific. ML is AI that learns from data instead of being explicitly programmed. DL is ML that uses multi-layer neural networks instead of simpler statistical models. GenAI is DL that creates new content instead of just classifying or predicting on existing data.

---

## 2. What is AI?

**Definition:** AI is the broad field of building machines that mimic human intelligence to perform tasks like problem-solving, decision-making, reasoning, perception, and language understanding.

**Examples:** Siri, self-driving cars, chess-playing bots, recommendation engines, spam filters.

**The key distinguishing idea:** AI systems can perform tasks autonomously — without needing a human to manually handle every decision — by applying some form of learned or programmed "intelligence."

```
Human-defined rules + data → AI system → autonomous task performance
```

**Backend engineer framing:** If a traditional program is `if-else` logic you wrote by hand, AI is a system that learns the `if-else` logic itself (or something far more sophisticated) from examples, instead of you hardcoding every rule.

---

## 3. Rule-Based AI

**Definition:** The earliest form of AI — systems that follow explicitly hardcoded rules written by humans. No learning from data involved.

```
IF temperature > 100°C THEN trigger_alarm()
IF email contains "lottery winner" THEN mark_as_spam()
```

**Characteristics:**
- Deterministic — same input always produces same output
- No data-driven learning — a human encodes every rule
- Brittle — fails on cases the rule-writer didn't anticipate
- Fast and fully explainable (you can trace exactly why a decision was made)

**Examples:** Early chess engines (minimax with hand-tuned heuristics), expert systems, basic chatbot decision trees, traditional fraud-detection rule engines.

**Why it matters historically:** Rule-based AI dominated from the 1950s–1990s before data-driven ML became practical (due to lack of compute and data). Understanding this gives context for why ML was such a breakthrough — it replaced "humans writing rules" with "machines discovering rules from data."

---

## 4. Machine Learning (ML)

**Definition:** ML is a subset of AI where machines learn patterns from data to improve their performance without being explicitly programmed for every scenario.

**How it works:** The system is shown data (often labeled examples), it identifies statistical patterns, and uses those patterns to make predictions on new, unseen data.

```
Historical Data → Algorithm finds patterns → Model → Predictions on new data
```

**Examples:** Spam detection, fraud detection, credit scoring, demand forecasting, recommendation systems.

**ML provides statistical tools to perform:**
- Statistical analysis
- Visualization
- Prediction
- Forecasting based on data

**Categories of ML:**

| Type | What it does | Example |
|---|---|---|
| Supervised Learning | Learns from labeled data (input → known output) | Spam detection (email → spam/not spam) |
| Unsupervised Learning | Finds patterns in unlabeled data | Customer segmentation |
| Reinforcement Learning | Learns via trial, error, and reward signals | Game-playing agents, robotics |

**Common algorithms (Traditional ML):** Linear Regression, Logistic Regression, Decision Trees, Random Forest, SVM, K-Means, XGBoost.

**Why ML matters for an AI Engineer:** You won't typically be building ML models from scratch in a GenAI engineering role, but understanding ML fundamentals helps you reason about model behavior, evaluation metrics, and why certain architectural choices (like neural networks for LLMs) were necessary.

---

## 5. Deep Learning (DL)

**Definition:** DL is a subset of ML that uses neural networks with multiple layers to analyze data, loosely inspired by the structure of the human brain.

**Correction from your notes (an important nuance):** DL is *inspired by* the brain's structure in a loose, abstract sense (layers of interconnected "neurons" that activate based on weighted inputs) — but it does not literally mimic how biological brains work. This is a common oversimplification worth being precise about in interviews.

```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → ... → Hidden Layer N → Output Layer
```

Each layer transforms the data, extracting increasingly abstract features. Early layers might detect edges in an image; later layers detect shapes; even later layers detect entire objects.

**Why DL needed more data and compute than traditional ML:** Multi-layer networks have millions to billions of parameters (weights) that need to be learned — this requires far more data and far more computation than simpler ML algorithms like Decision Trees or Logistic Regression.

**Examples:** Face recognition, speech recognition, machine translation, image generation.

---

## 6. Neural Network Architectures: ANN, CNN, RNN, Transformers

This is the layer of detail your notes hinted at (ANN, CNN, RNN, Transformers) but didn't fully expand — here's the complete picture.

```
Deep Learning
│
├── ANN (Artificial Neural Network)
│      → general-purpose, fully-connected layers
│      → foundation for all other architectures
│
├── CNN (Convolutional Neural Network)
│      → specialized for spatial data (images, video)
│      → used in: Computer Vision (object detection, image classification)
│
└── RNN (Recurrent Neural Network) and its variants (LSTM, GRU)
       → specialized for sequential data (text, time series)
       → used in: text-related tasks, time series forecasting
       │
       └── Transformers and BERT (the evolution beyond RNN)
              → solved RNN's sequential bottleneck using self-attention
              → foundation of every modern LLM (GPT, Claude, Llama, Gemini)
```

**ANN (Artificial Neural Network):** The most basic neural network — layers of neurons fully connected to the next layer. Good for tabular data and simple pattern recognition, but doesn't exploit any special structure in the data.

**CNN (Convolutional Neural Network):** Uses sliding filters (convolutions) to detect local spatial patterns — edges, textures, shapes. This makes CNNs extremely effective for images because nearby pixels are related to each other in a way a fully-connected ANN can't efficiently exploit.

**RNN (Recurrent Neural Network):** Processes sequences one element at a time, maintaining a "memory" (hidden state) of what it has seen so far. Designed for text, speech, and time series — anything where order matters. Limitation: struggles with long sequences due to vanishing gradients (forgets early context).

**Transformers:** Replaced RNNs as the dominant architecture for sequential data starting in 2017 ("Attention Is All You Need" paper). Instead of processing tokens one at a time, Transformers use self-attention to let every token look at every other token simultaneously — solving RNN's long-range memory problem and enabling massive parallelization during training. Every modern LLM (GPT, Claude, Gemini, Llama) is Transformer-based.

**BERT:** A Transformer-based model (encoder-only) designed for understanding tasks like classification and search — not text generation. Worth knowing as a contrast to GPT-style (decoder-only) generative models.

---

## 7. Discriminative vs Generative Models

**This distinction is foundational to understanding where GenAI sits in the model landscape.**

```
                    Models
                       │
        ┌──────────────┴──────────────┐
        │                              │
  Discriminative                  Generative
        │                              │
      Task                           Task
        │                              │
  Classification,              Generate new data,
  Prediction                   trained on huge datasets
        │                              │
  Trained on                          │
  labeled dataset              ┌──────┴──────┐
                               LLM            LIM
                            (text)          (image)
```

**Discriminative Models:**
- Learn to distinguish between classes/categories
- Answer the question: "Given this input, what category does it belong to?"
- Trained on labeled datasets (supervised learning)
- Examples: spam classifier, fraud detector, sentiment analyzer

**Generative Models:**
- Learn the underlying distribution of the data itself
- Answer the question: "What does data like this typically look like? Now generate something new like it."
- Trained on huge (often unlabeled or self-labeled) datasets
- Examples: LLMs (generate text), Image generation models (generate images), Music generation models

**Why this distinction matters for an AI Engineer:** GenAI's entire value proposition is built on generative models. Traditional ML and most of "classical AI" work was discriminative (classify, predict, detect). GenAI flipped the paradigm to "create new content that didn't exist before."

---

## 8. Generative AI (GenAI)

**Definition:** GenAI is AI that creates new content — text, images, music, code, video — based on patterns learned from training data, rather than just classifying or predicting on existing data.

```
GenAI = Subset of Deep Learning that performs the Generative task
```

**Examples:** ChatGPT, OpenAI's models, Gemini, Claude (text generation), DALL-E, Midjourney (image generation).

**What makes GenAI different from earlier AI:**

| Earlier AI/ML | GenAI |
|---|---|
| Classifies or predicts existing categories | Creates entirely new content |
| Output: a label or number | Output: text, image, audio, video, code |
| "Is this email spam?" | "Write me an email" |
| Trained on labeled data, narrow task | Trained on massive, broad data, general capability |

**Sub-fields within GenAI (from your hierarchy diagram):**
- LLM — text generation and understanding
- RAG — combining retrieval with generation for grounded answers
- AI Agents — systems that reason and act autonomously
- Multi-Modal AI — models that handle multiple data types at once

---

## 9. NLP (Natural Language Processing)

**Definition:** NLP is the AI field focused on understanding and processing human language — both written and spoken.

**Examples:** Translation, sentiment analysis, named entity recognition, text classification.

**Common NLP tools and libraries:** NLTK, spaCy, Transformers (Hugging Face library, not to be confused with the Transformer architecture itself).

**Where NLP sits relative to LLMs:** LLMs are the modern, dominant approach to solving NLP tasks. Before LLMs, NLP required separate specialized models for each task (one model for translation, another for sentiment, another for NER). LLMs collapsed most of these into a single general-purpose model that can do all of them through prompting.

```
Pre-LLM NLP:  Translation model + Sentiment model + NER model + Summarization model
                (4 separate trained models)

LLM-era NLP:  One LLM + different prompts
                "Translate this" / "What's the sentiment?" / "Extract entities" / "Summarize this"
```

---

## 10. Computer Vision

**Definition:** The AI field focused on understanding and interpreting images and videos.

**Examples:** Object detection in CCTV footage, facial recognition, medical imaging analysis, autonomous vehicle perception.

**Underlying architecture:** Traditionally CNN-based; increasingly Vision Transformers (ViT) and multi-modal models are taking over state-of-the-art performance.

**Use cases relevant to an AI Engineer:**
- Resume screenshot parsing / document OCR
- Document analysis (extracting structured data from scanned PDFs)
- Medical imaging (diagnostic assistance)
- Receipt/invoice processing pipelines

---

## 11. Large Language Models (LLM)

**Definition:** An LLM is a Generative AI model trained on massive amounts of text data to understand and generate human-like language.

```
Input Text → LLM → Generated Response
```

**Examples:** GPT-4, Claude, Gemini, Llama.

**The core mechanism — LLMs are next-token prediction machines:**

The model predicts the most likely next token (a word or word-piece) given everything that came before it. It repeats this process thousands of times per second until a complete response is generated.

```
"The capital of France is" → predicts → "Paris"
"The capital of France is Paris" → predicts → "."
"The capital of France is Paris." → predicts → <EOS> (stop)
```

This single mechanism — repeated next-token prediction — is the entire generative process underlying every LLM response, no matter how complex or "intelligent" it appears.

---

## 12. Why Is It Called "Large"?

LLMs are called "large" because of two dimensions of scale:

**1. Billions of words (training data):** Trained on text scraped from books, websites, code repositories, articles — effectively a large fraction of publicly available text on the internet.

**2. Billions of parameters (weights):** The learned numerical values inside the neural network that encode everything the model "knows." More parameters generally means better understanding and reasoning ability, up to a point (with diminishing returns and rising cost).

```
More Parameters → More capacity to learn complex patterns
                → Better understanding and reasoning (generally)
                → But also: more compute cost, slower inference
```

**Rough scale reference:**
```
GPT-2:        1.5 billion parameters
GPT-3:        175 billion parameters
GPT-4:        rumored ~1.7 trillion parameters (mixture of experts)
Llama 3 70B:  70 billion parameters
```

---

## 13. Why Is It Called a "Language Model"?

Because it works specifically with **language** — text — performing tasks like:

- Text generation
- Summarization
- Translation
- Question answering
- Code generation
- Chatbots / conversational AI

A "model" in the ML sense is just a mathematical function with learned parameters that maps an input to an output. A "language model" specifically maps sequences of text to predictions about more text (most commonly, the next token).

---

## 14. How an LLM Generates a Response — The Flow

This is the complete request-to-response pipeline, matching your notes' flow diagram, expanded with what happens at each stage:

```
Your Question (raw text input)
        ↓
Tokenization (text broken into tokens, converted to token IDs)
        ↓
LLM Processing (token IDs → embeddings → Transformer layers → 
                self-attention computes relationships between tokens)
        ↓
Predict Next Token (final layer outputs probability distribution 
                     over the entire vocabulary)
        ↓
[Repeat: append predicted token, predict the next one, repeat thousands of times]
        ↓
Generate Response (token sequence converted back to human-readable text)
```

**Backend engineer analogy:** Think of this as a request-response cycle, except the "response" is built incrementally — one token at a time — like a streaming API response, rather than computed all at once and returned.

---

## 15. Foundation Models vs Fine-Tuned Models

**Foundation Models** are large, general-purpose models pre-trained on massive, diverse datasets. They can perform many tasks reasonably well out of the box, without task-specific training.

```
Foundation Model = Pre-trained on huge, general data
                  = General-purpose, broad capability
                  = Examples: GPT-4, Claude 3, Gemini, Llama
```

**Fine-Tuned Models** take a foundation model and continue training it on a smaller, specific dataset to specialize it for a particular use case or domain.

```
Foundation Model
       ↓
Fine-Tuning on domain-specific data
       ↓
Fine-Tuned Model (specialized)

Example:
GPT (foundation model)
       ↓
Trained on medical data
       ↓
Medical Assistant (fine-tuned model)
```

**Other real-world fine-tuning examples:** Legal document assistant (fine-tuned on legal text and case law), customer support bot (fine-tuned on a company's support tickets and documentation).

**Tools commonly used for fine-tuning:** Hugging Face (model hub + training libraries), LoRA (Low-Rank Adaptation — efficient fine-tuning technique), PEFT (Parameter-Efficient Fine-Tuning — a broader category that includes LoRA).

**When to fine-tune vs when to use RAG:** This is one of the most common AI Engineer interview questions. The short answer: fine-tune when you need to change the model's *behavior, style, or format*; use RAG when you need the model to *know* information it wasn't trained on. Most production systems use RAG far more often than fine-tuning because it's cheaper, faster to update, and doesn't risk degrading the base model's general capabilities.

---

## 16. Major LLM Providers and Models

```
OpenAI          Meta            Google           Anthropic
   │              │                │                  │
  GPT-4         Llama           Gemini            Claude 3
  GPT-3.5       Llama 2         Gemma             Claude 4
                                                       │
                                              All of these are
                                              Foundation Models
                                              (Pre-trained on huge,
                                               domain-general data)
```

| Company | Model Family | Notes |
|---|---|---|
| OpenAI | GPT-4, GPT-3.5, GPT-5, o1/o3 (reasoning models) | Most widely integrated via API |
| Meta | Llama 2, Llama 3, Llama 4 | Open-weight, widely used for self-hosting and fine-tuning |
| Google | Gemini, Gemma | Gemini = closed/API, Gemma = open-weight smaller models |
| Anthropic | Claude 3, Claude 4 | Strong focus on safety, long context, instruction following |
| Mistral AI | Mistral 7B, Mixtral | Open-weight, strong performance-to-size ratio |
| DeepSeek | DeepSeek-V3, DeepSeek-R1 | Open-weight, strong reasoning models |

**Foundation models → Pre-trained models → trained further (fine-tuning) → domain-specific use case** — this is the lifecycle from your notes' diagram, and it's worth memorizing as a single chain for interviews.

---

## 17. What an LLM Cannot Do Alone

This is one of the most important conceptual sections for an AI Engineer to internalize — it's the entire justification for why the rest of the GenAI stack (RAG, agents, vector DBs) exists.

**An LLM by itself:**
- Doesn't know your company's internal documents (it was trained on public data only, up to a cutoff date)
- Doesn't know today's latest data (unless connected to live tools — web search, APIs)
- Doesn't access a database automatically (no built-in ability to query SQL or any data store)
- Doesn't have long-term memory (each API call is stateless — it has no memory of past conversations unless you explicitly provide that history again)

**That's exactly why we use the following stack to build real applications:**

```
LLM
 ↓
Embeddings   (convert text into searchable numerical vectors)
 ↓
Vector DB    (store and search those vectors efficiently)
 ↓
RAG          (retrieve relevant info and feed it to the LLM as context)
 ↓
Agents       (let the LLM reason, choose tools, and take actions)
 ↓
→ Real AI Applications
```

**The single most important mental model for your career transition:**

As an AI Engineer, your job is to build applications *around* LLMs — not to create the models themselves. Model training is a research/ML engineering specialty (done by OpenAI, Anthropic, Google, Meta). Your job is closer to traditional backend/software engineering: integrating LLM APIs, designing data pipelines, building retrieval systems, and orchestrating multi-step agent workflows — using the skills you already have (APIs, databases, system design) applied to a new kind of "compute" (the LLM).

---

## 18. Embeddings

**Definition:** Embeddings are numerical vector representations of text, images, or other data that capture semantic meaning — not just surface-level characters, but the underlying meaning of the content.

```
"I love programming" → [0.21, -0.45, 0.78, 0.12, ... ] (a vector of hundreds/thousands of numbers)
```

**The key property:** Texts with similar meaning produce vectors that are mathematically close to each other in vector space, even if they share no exact words.

```
"I love coding"        → vector close to → "Programming is my passion"
"I love coding"        → vector far from → "The weather is sunny today"
```

**Examples / use cases:** Finding similar resumes (HR tech), semantic search (search by meaning, not keyword match), RAG (retrieving relevant document chunks), recommendation systems, duplicate detection.

**Popular embedding models:** OpenAI's `text-embedding-3-small` and `text-embedding-3-large`, Cohere `embed-v3`, open-source options like `BGE` and `E5` from Hugging Face.

**Why embeddings matter for an AI Engineer:** Embeddings are the foundation of every semantic search and RAG system you'll build. Understanding that "similarity" in AI systems is measured as distance/angle between vectors (not string matching) is fundamental.

---

## 19. Vector Databases

**Definition:** A vector database is a database specifically optimized for storing and searching embeddings — finding the most similar vectors to a given query vector, extremely fast, even across millions or billions of entries.

**Examples / use cases:** Semantic document search, RAG pipelines, recommendation engines, duplicate/anomaly detection.

**Popular vector databases:**

| Database | Type | Notes |
|---|---|---|
| FAISS | Library (Meta) | Fast, runs locally, great for prototyping and research |
| Pinecone | Managed cloud service | Fully managed, popular in production RAG systems |
| Weaviate | Open-source / cloud | Supports hybrid search (vector + keyword) |
| ChromaDB | Open-source | Lightweight, popular for local development and small-to-medium projects |
| Qdrant | Open-source / cloud | High performance, strong metadata filtering |
| pgvector | PostgreSQL extension | Good option if you already run Postgres — adds vector search to existing infra |

**Why a vector DB instead of a regular SQL database:** Traditional databases excel at exact-match queries (`WHERE id = 5`). Vector databases excel at "find me the most similar things" queries — comparing the mathematical distance between vectors using metrics like cosine similarity. This requires specialized indexing algorithms (like HNSW) that SQL databases don't natively support efficiently at scale (though `pgvector` brings this capability into Postgres).

---

## 20. RAG (Retrieval-Augmented Generation)

**Definition:** RAG is a technique that combines retrieving external data with an LLM to generate accurate, grounded answers — instead of relying purely on what the LLM memorized during training.

```
User Query
    ↓
Embed the query → search vector DB → retrieve relevant chunks
    ↓
Inject retrieved chunks into the LLM's prompt as context
    ↓
LLM generates an answer based on that retrieved context
    ↓
Grounded, accurate response (not just "memorized" knowledge)
```

**Examples / use cases:** Chat with PDF documents, internal company knowledge base assistants, customer support bots grounded in product documentation.

**Common tools in the RAG ecosystem:** LangChain, LlamaIndex, ChromaDB, Pinecone.

**Why RAG exists — the problem it solves:** LLMs have a fixed knowledge cutoff and don't know your private/internal data. RAG solves this without retraining the model — you simply retrieve the relevant facts at query time and hand them to the LLM as context, the same way you'd pass parameters into a function call.

**This is probably the single most commonly built system in AI Engineering roles today** — expect deep interview questions on chunking strategy, embedding model choice, retrieval quality, and how to handle cases where retrieval returns irrelevant or insufficient context.

---

## 21. AI Agents

**Definition:** An AI Agent is an AI system that can reason, use tools, and perform actions autonomously to achieve a goal — going beyond simply answering a question to actually *doing* something.

```
User Goal
    ↓
Agent reasons: "What do I need to do to achieve this goal?"
    ↓
Agent selects and calls a tool (API, database, search, calculator, etc.)
    ↓
Agent observes the result
    ↓
Agent decides: "Is the goal achieved, or do I need another step?"
    ↓
[Repeat until goal is achieved]
    ↓
Final outcome / response
```

**Examples / use cases:** An AI that books a flight automatically (searches, compares prices, completes the booking), a coding agent that writes, tests, and debugs code in a loop, a research agent that searches the web, reads multiple sources, and synthesizes a report.

**Popular agent frameworks:** LangGraph, CrewAI, AutoGen, OpenAI's Agents SDK.

**Why agents are different from a simple LLM call:** A plain LLM call is a single request → single response. An agent involves a *loop* — the LLM decides what action to take, an external system executes that action, the result is fed back to the LLM, and this continues until the task is complete. This is the architectural leap from "chatbot" to "autonomous system."

---

## 22. Agentic AI

**Definition:** Agentic AI refers to AI-built applications that can perform their own tasks without requiring human intervention at every step — making decisions, taking sequences of actions, and adapting based on results, autonomously.

**Examples:** Netflix's recommendation system (continuously adapts without a human manually tuning every recommendation), self-driving cars (perceive, decide, and act in real time without human input for each decision).

**Agentic AI vs AI Agents — a subtle but important distinction for interviews:**

| | AI Agent | Agentic AI |
|---|---|---|
| Scope | A specific system/implementation (e.g., one agent that books flights) | A broader paradigm/characteristic — describes *how* an application behaves |
| Usage | "We built an AI agent for customer support" | "This is an agentic system — it decides and acts autonomously" |
| Relationship | A concrete instance | The general property that instance exhibits |

In practice, the terms are often used interchangeably in industry conversation, but "Agentic AI" tends to describe the broader design philosophy (autonomy, multi-step reasoning, tool use, minimal human-in-the-loop), while "AI Agent" refers to a specific implemented system exhibiting that philosophy.

---

## 23. Multi-Modal AI

**Definition:** AI that can understand and process multiple data types simultaneously — such as text, images, audio, and video — within a single model.

```
Input: [Text] + [Image] + [Audio]
              ↓
        Multi-Modal Model
              ↓
        Unified Understanding / Response
```

**Examples / use cases:** Upload an image and ask questions about it ("What's wrong with this X-ray?"), GPT-4o, Gemini, Claude (all support text + image input natively).

**Why multi-modal matters:** Real-world problems rarely come in a single data format. A customer support ticket might include a screenshot; a medical case might include both patient notes (text) and scan images. Multi-modal models eliminate the need to stitch together separate specialized models (one for vision, one for text) and instead reason jointly across modalities.

---

## 24. Types of AI Models by Modality

This section consolidates and expands the "Types of Models" portion of your notes into one clear reference table.

| Model Type | What It Does | Use Cases | Examples |
|---|---|---|---|
| **LLM** | Works with text | Chatbots, code generation, summarization, Q&A | GPT-4, Claude, Gemini, Llama |
| **Embedding Models** | Convert text into vectors | Semantic search, RAG, similarity matching | OpenAI `text-embedding-3-small/large`, Cohere embed-v3 |
| **Image Generation Models** | Generate images from text | Content creation, marketing, design | DALL-E, Midjourney, Stable Diffusion |
| **Vision Models** | Understand images | Resume screenshot parsing, OCR, document analysis, medical imaging | GPT-4V, Gemini Vision, Claude Vision |
| **Speech Models** | Convert speech to text (and vice versa) | Voice assistants, transcription | Whisper (OpenAI), ElevenLabs |
| **Multi-Modal Models** | Handle multiple input types at once | Image + text Q&A, document processing, AI assistants | GPT-4o, Gemini, Claude |
| **Foundation Models** | General-purpose, broad capability | Can perform many tasks out of the box | GPT, Claude, Gemini, Llama |
| **Fine-Tuned Models** | Foundation model specialized for a specific task | Domain-specific assistants (legal, medical, support) | Custom fine-tuned GPT/Llama variants |

---

## 25. Prompt Engineering

**Definition:** Prompt engineering is the practice of designing effective instructions to get better, more accurate, and more consistent responses from AI models.

**Example:** "Summarize this article in 5 points" — rather than just "summarize this," which gives the model less structure to follow and produces inconsistent output length/format.

**Core techniques an AI Engineer should know:**

| Technique | What it does |
|---|---|
| Zero-shot prompting | Ask directly with no examples |
| Few-shot prompting | Provide 2–5 examples of input → output to set the format/pattern |
| Chain-of-Thought (CoT) | Ask the model to reason step-by-step before answering — improves accuracy on complex tasks |
| Role prompting | Assign the model a persona/role ("You are a senior Python code reviewer...") |
| Structured output prompting | Instruct the model to respond in JSON/XML/specific schema for reliable parsing downstream |

**Why this matters in production:** Prompt engineering isn't just about getting a "nicer" answer — in production systems, prompt design directly impacts output consistency, parseability (can your code reliably extract structured data from the response?), token cost, and latency.

---

## 26. Fine-Tuning

**Definition:** Fine-tuning is the process of further training a pre-trained (foundation) model on custom, domain-specific data to specialize it for a particular use case.

```
Pre-trained Foundation Model
            ↓
Train further on custom/domain-specific dataset
            ↓
Fine-Tuned Model (specialized for your use case)
```

**Examples:** A legal document assistant fine-tuned on contracts and case law, a customer support model fine-tuned on a company's historical support tickets.

**Key tools and techniques:**

| Tool/Technique | What it is |
|---|---|
| Hugging Face | The dominant open-source hub and library ecosystem for hosting and fine-tuning models |
| LoRA (Low-Rank Adaptation) | An efficient fine-tuning method that trains small "adapter" matrices instead of updating all model weights — drastically reduces compute and memory needed |
| PEFT (Parameter-Efficient Fine-Tuning) | The broader category of techniques (including LoRA) that fine-tune a model by updating only a small fraction of its parameters |
| QLoRA | LoRA combined with quantization — enables fine-tuning large models on consumer-grade GPUs |

**When to choose fine-tuning over RAG (important interview distinction):**

```
Fine-tuning → use when you need to change BEHAVIOR, TONE, FORMAT, or teach a 
              specialized SKILL the base model lacks (e.g., always respond in 
              a specific JSON schema, or write in a particular legal style)

RAG         → use when you need the model to KNOW specific facts/information 
              it wasn't trained on (e.g., your company's product documentation, 
              today's news, internal policies)
```

In most real-world GenAI engineering roles, you'll reach for RAG far more often than fine-tuning, because RAG is cheaper, doesn't require ML training infrastructure, and is trivially updatable (just re-index your documents) — whereas fine-tuning requires a retraining cycle every time the underlying knowledge changes.

---

## 27. MLOps

**Definition:** MLOps is the set of practices for developing, deploying, monitoring, and managing ML/AI models in production — the DevOps equivalent for machine learning systems.

```
Model Development → Training → Validation → Deployment → Monitoring → Retraining
        (this is a continuous cycle, not a one-time process)
```

**Examples:** Automated model deployment pipelines, A/B testing different model versions in production, automatic retraining when model performance degrades (model drift).

**Popular MLOps tools:**

| Tool | Purpose |
|---|---|
| MLflow | Experiment tracking, model versioning, deployment |
| Kubeflow | Kubernetes-native ML pipeline orchestration |
| SageMaker | AWS's managed ML platform (training, deployment, monitoring) |
| Weights & Biases (W&B) | Experiment tracking and visualization |

**Why MLOps matters even for a GenAI-focused engineer:** While you may not be training models from scratch, you will still need to think about versioning prompts, tracking which model version produced which output, monitoring for quality degradation, and managing rollouts of new model versions — all MLOps concepts adapted to the GenAI context (sometimes called LLMOps, see below).

---

## 28. LLMOps (the GenAI-Specific Layer)

While your notes covered MLOps generally, it's worth explicitly distinguishing **LLMOps** — the subset of operational practices specific to LLM-based applications, since this is what you'll actually be doing day-to-day as an AI Engineer.

```
LLMOps covers:

Prompt versioning & management   → track which prompt version produced which output
Token usage & cost monitoring    → track spend per endpoint/user/feature
Latency monitoring                → especially important for streaming responses
Output quality evaluation         → RAGAS, LLM-as-judge, human eval pipelines
Guardrails & safety filtering     → prevent harmful/off-topic outputs
A/B testing prompts/models        → compare GPT-4o-mini vs Claude Haiku on the same task
Caching                           → reduce redundant LLM calls and cost
```

**Popular LLMOps tools:** LangSmith (LangChain's observability platform), Langfuse (open-source LLM observability), Helicone (proxy-based LLM monitoring), Arize Phoenix.

**Why this distinction matters:** MLOps deals with traditional model training/deployment lifecycles. LLMOps deals with the unique challenges of *using* someone else's pre-trained model via API — cost per token, prompt drift, hallucination monitoring — concerns that didn't exist in classical MLOps.

---

## 29. LangChain, LangGraph, LlamaIndex

These are the three most important frameworks an AI Engineer needs to know, and your notes correctly grouped them under "AI Engineer Perspective" tools.

**LangChain:** A general-purpose framework for building LLM-powered applications. Provides standardized abstractions for prompts, chains (sequences of LLM calls), memory, document loaders, and integrations with vector databases and tools.

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_template("Explain {concept} simply")
chain = prompt | llm
chain.invoke({"concept": "vector embeddings"})
```

**LlamaIndex:** Specialized specifically for data ingestion, indexing, and retrieval — i.e., it's the most RAG-focused of the three frameworks. Often preferred over LangChain specifically for production-grade RAG pipelines due to cleaner abstractions for document processing.

**LangGraph:** Built on top of LangChain, designed for building stateful, graph-based agent workflows — where the agent's logic has branches, loops, and multiple decision points (not just a linear chain of LLM calls).

```
LangChain  → best for: prototyping, standard RAG, chaining LLM calls
LlamaIndex → best for: production RAG, document-heavy pipelines, complex indexing
LangGraph  → best for: complex agents with branching logic, multi-step workflows
```

**Why all three matter:** You'll likely use a combination of these in real projects — LlamaIndex or LangChain for the RAG/retrieval layer, and LangGraph when the application needs agent-like decision-making rather than a single linear pipeline.

---

## 30. Model Context Protocol (MCP)

**Definition:** MCP is an open standard (introduced by Anthropic) that allows AI models to connect to external tools and data sources through a single, standardized interface — instead of building a custom integration for every tool.

**Analogy:** MCP is like "USB-C for AI tools" — one standard port that any compatible tool can plug into, instead of every tool needing its own proprietary connector.

```
Without MCP:
LLM → custom integration → Database
LLM → custom integration → GitHub
LLM → custom integration → Slack
(each integration is bespoke and doesn't transfer to other projects)

With MCP:
LLM → MCP Client → MCP Protocol → MCP Servers (Database, GitHub, Slack, etc.)
(standardized — write the integration once, reuse it anywhere)
```

**Why this is relevant to an AI Engineer today (2025–2026):** MCP has rapidly become the standard way agentic systems connect to external tools and data sources. Claude Desktop, Claude Code, and an increasing number of IDEs and platforms support MCP natively. Understanding MCP server/client architecture is increasingly expected in AI Engineering interviews, especially for agent-focused roles.

---

## 31. Hallucinations

**Definition:** Hallucinations occur when an LLM generates confident, fluent, but factually incorrect information — because the model is fundamentally predicting the most *statistically plausible* next token, not verifying *truth* against any ground source.

```
Model has no built-in fact-checking mechanism
        ↓
It generates text based on learned patterns of what "sounds right"
        ↓
If the pattern is plausible but factually wrong → hallucination
```

**Why this matters directly for the systems you'll build:** RAG exists largely as a hallucination-mitigation strategy — by grounding the model's answer in retrieved factual context, you reduce (but don't eliminate) the chance of fabricated information. Other mitigation strategies include lowering temperature (more deterministic output), explicit prompting to say "I don't know" when uncertain, and self-consistency checks (sampling multiple responses and checking agreement).

---

## 32. Context Window

**Definition:** The context window is the maximum number of tokens (input + output combined) an LLM can process in a single interaction — its "working memory" for that specific call.

```
Context Window = System Prompt + Conversation History + 
                  Retrieved Documents + User Message + Model's Output
```

**Why it matters for system design:** Every component you inject into a prompt (retrieved RAG chunks, conversation history, tool definitions) competes for the same fixed token budget. Designing what to include — and what to leave out or summarize — is one of the most practical day-to-day engineering challenges in building LLM applications.

**Quick reference of typical context windows:**

| Model | Context Window |
|---|---|
| GPT-4o | 128,000 tokens |
| Claude 3.5/4 Sonnet | 200,000 tokens |
| Gemini 1.5/2.0 Pro | up to 1,000,000+ tokens |

---

## 33. Tokens & Tokenization (Quick Primer)

**Definition:** A token is the smallest unit of text an LLM processes — not always a full word. Tokenization is the process of breaking raw text into these units before feeding it to the model.

```
"unbelievable" → ["un", "believ", "able"]   (3 tokens, 1 word)
```

**Why this matters practically:** API pricing is based on token count (both input and output), so understanding tokenization directly affects cost estimation, prompt design, and context window budgeting — all daily concerns for an AI Engineer.

*(This topic deserves its own deep-dive document — tokens, context windows, and API costing are covered in much greater depth in a separate companion file if you want the full breakdown.)*

---

## 34. Open Source vs Closed Source Models

Your notes touched on this distinction — worth making explicit since it's a common interview and architectural-decision topic.

| | Open Source / Open-Weight Models | Closed Source Models |
|---|---|---|
| Examples | Llama, Mistral, Gemma, DeepSeek | GPT-4, Claude, Gemini |
| Access | Download weights, self-host | API access only |
| Customization | Full fine-tuning control | Limited to prompting / API-level fine-tuning |
| Cost model | Infrastructure cost (your own GPUs/cloud) | Pay-per-token API pricing |
| Data privacy | Full control — data never leaves your infra | Data sent to provider's servers (subject to their policies) |
| Maintenance burden | You manage hosting, scaling, updates | Provider manages everything |

**When to choose which (a common system design interview question):** Closed-source APIs (GPT, Claude, Gemini) are the default choice for most production applications due to ease of integration and best-in-class quality. Open-source/open-weight models become attractive when you have strict data privacy requirements (data cannot leave your infrastructure), need to fine-tune extensively for a narrow domain, or are operating at a scale where self-hosting becomes more cost-effective than per-token API pricing.

---

## 35. Traditional ML vs GenAI — Side by Side

This comparison ties together the entire hierarchy and is worth having crisp for interviews, since it's often asked directly as "what's the difference between your ML background and what you'll be doing as an AI Engineer?"

| | Traditional ML | GenAI |
|---|---|---|
| Output | A label, number, or category | New content (text, image, audio, code) |
| Model type | Discriminative (mostly) | Generative |
| Training data | Often labeled, task-specific | Massive, broad, often self-supervised |
| Typical task | Classify, predict, forecast | Create, generate, converse, reason |
| Your role as engineer | Build/train models, feature engineering | Integrate pre-trained models via API, build retrieval/agent systems around them |
| Example | "Will this customer churn?" | "Write a churn-prevention email for this customer" |

---

## 36. The AI Engineer's Toolkit — Complete Map

This consolidates the "AI Engineer Perspective" section from your notes into the full, organized toolkit you'll actually touch in this role.

```
Core Model Layer
├── LLM            — the reasoning/generation engine (GPT, Claude, Gemini, Llama)
├── Embedding       — converts text/data into searchable vectors
└── Vision Models   — image/document understanding

Knowledge & Retrieval Layer
├── Vector DB       — FAISS, Pinecone, ChromaDB, Weaviate, Qdrant
└── RAG             — retrieval + generation pipeline

Orchestration Layer
├── LangChain       — general LLM app framework
├── LlamaIndex      — RAG-focused data framework
└── LangGraph       — stateful, branching agent workflows

Action Layer
├── Agents          — autonomous, tool-using systems
└── MCP             — standardized tool/data connectivity

Access Layer
└── OpenAI APIs / Anthropic APIs / Google APIs — the actual model access points

Operations Layer
├── Prompt Engineering  — designing effective inputs
├── Fine-Tuning         — specializing models (LoRA, PEFT, Hugging Face)
├── LLMOps              — monitoring, cost tracking, evaluation, observability
└── MLOps               — broader deployment/monitoring practices (MLflow, SageMaker)
```

---

## 37. The AI Engineer's Job — What It Actually Is

This is the single most important mental shift for your career transition, and it's the note you wrote yourself on page 9 — worth elevating to its own closing section because it reframes everything above.

**As an AI Engineer, your job is to build applications around LLMs — not to create the models themselves.**

```
Model Training / Research          →  Done by OpenAI, Anthropic, Google, Meta
       (Research Scientists, ML Engineers — deep math/ML theory, massive compute)

Application Engineering            →  Your role as an AI Engineer
       (Backend/software engineering skills applied to LLM-based systems —
        APIs, databases, system design, retrieval pipelines, agent orchestration)
```

**What this means practically for your transition:** Your existing skills — Python, REST APIs, PostgreSQL, ETL pipelines, system design, debugging, testing — are not being replaced. They're being redirected. Instead of building a CRUD API that talks to a relational database, you'll build an API that talks to an LLM, retrieves context from a vector database, and orchestrates multi-step agent logic. The underlying engineering discipline (clean code, testing, observability, scalability) is identical — the new "external service" you're integrating with happens to be a language model instead of a payment gateway or a third-party SaaS API.

This is precisely why your backend background is a genuine asset, not a deficit, in this transition — you already know how to build robust systems around external services. The LLM is simply a new (and very powerful) kind of external service.
