# LLM Foundations and Internal Working
### A First-Principles Guide for Backend Engineers Transitioning to AI Engineering

> **How to use this document:** Read top-to-bottom on your first pass — each section builds on the previous. Use the Table of Contents for targeted revision. Every concept answers: *Why does this exist? What problem does it solve? How does it connect to the bigger picture?*

---

## Table of Contents

1. [The Big Picture: How an LLM Processes Text](#1-the-big-picture)
2. [Why Transformers Replaced RNNs and CNNs](#2-why-transformers)
3. [Tokenization](#3-tokenization)
4. [Vocabulary and Token IDs](#4-vocabulary-and-token-ids)
5. [Embeddings](#5-embeddings)
6. [Positional Encoding](#6-positional-encoding)
7. [Self-Attention and Q, K, V](#7-self-attention-and-qkv)
8. [Multi-Head Attention](#8-multi-head-attention)
9. [The Transformer Block](#9-the-transformer-block)
10. [Linear Layer + Softmax → Next Token Prediction](#10-linear-layer--softmax)
11. [Detokenization](#11-detokenization)
12. [The Training Phase](#12-the-training-phase)
13. [Self-Supervised Learning](#13-self-supervised-learning)
14. [What "Pre-Trained" Means](#14-what-pre-trained-means)
15. [Encoder vs Decoder vs Encoder-Decoder](#15-encoder-vs-decoder)
16. [Fine-Tuning (Full, PEFT, LoRA)](#16-fine-tuning)
17. [Inference and Sampling Strategies](#17-inference-and-sampling-strategies)
18. [Context Window and Attention Limits](#18-context-window)
19. [Hallucinations — Why They Happen](#19-hallucinations)
20. [Chain of Thought Reasoning](#20-chain-of-thought-reasoning)
21. [Reasoning Models](#21-reasoning-models)
22. [Small Language Models (SLMs)](#22-small-language-models)
23. [Knowledge Distillation](#23-knowledge-distillation)
24. [Quantization](#24-quantization)
25. [BOS, EOS, and Special Tokens](#25-special-tokens)

---

## 1. The Big Picture

Before diving into components, understand the full pipeline. An LLM is a system that takes text in and produces text out — but underneath, everything is numbers.

```
Raw Text Input
     ↓
Tokenization          ← Break text into tokens
     ↓
Token IDs             ← Map tokens to integers
     ↓
Embeddings            ← Map integers to vectors (meaning)
     ↓
Positional Encoding   ← Add order information
     ↓
Transformer Layers    ← Understand relationships (stacked N times)
  ├── Self-Attention
  ├── Add & Normalize
  ├── Feed-Forward Network
  └── Add & Normalize
     ↓
Linear Layer          ← Score every vocabulary token
     ↓
Softmax               ← Convert scores to probabilities
     ↓
Next Token            ← Sample or pick the top token
     ↓
Detokenization        ← Convert token back to text
     ↓
Repeat until EOS
     ↓
Final Response
```

**The key insight:** LLMs are next-token predictors. Everything else — reasoning, summarization, code generation — emerges from doing this prediction extremely well across trillions of examples.

**Backend analogy:** Think of the pipeline like an HTTP request through a middleware stack. Each layer transforms the data and passes it forward. The model's weights are the business logic baked in during training.

---

## 2. Why Transformers

**The problem Transformers solved:** Before 2017, NLP used RNNs and CNNs. Both had critical limitations.

### RNNs (Recurrent Neural Networks)

```
Word1 → Word2 → Word3 → ... → Word100
```

RNNs process text **sequentially** — each word depends on the previous. To understand word 100, the model must carry information through 99 steps.

**Problems:**
- Sequential processing → cannot parallelize → slow training
- Vanishing gradients → information from early tokens fades
- Poor long-range memory → "The animal didn't cross the road because **it** was too tired" — an RNN may forget what "it" refers to by the time it reaches the end

### CNNs (Convolutional Neural Networks)

CNNs use fixed-size local windows to detect patterns. They parallelize well but cannot capture relationships between words that are far apart — a CNN would need many layers to connect distant tokens.

### Transformers (2017 — "Attention Is All You Need")

Google's landmark paper introduced **Self-Attention**: every token can directly attend to every other token in a single step.

```
Word1 <--> Word2
Word1 <--> Word3
Word1 <--> Word100   ← direct, no intermediate steps
```

**Three transformative advantages:**

| Feature | RNN | CNN | Transformer |
|---|---|---|---|
| Sequential Processing | Yes | No | No |
| Parallel Processing | No | Yes | Yes |
| Long-Range Dependencies | Poor | Moderate | Excellent |
| Training Speed | Slow | Fast | Fast |
| Scalability | Low | Medium | High |
| Used in Modern LLMs | No | No | Yes |

**Why it matters for you:** All modern LLMs (GPT, Llama, Claude, Gemini, Mistral) are Transformer-based. Understanding the Transformer is the single most important foundation for AI Engineering.

---

## 3. Tokenization

**What it is:** The process of converting raw text into smaller units (tokens) that the model can process.

**Why it exists:** Neural networks operate on numbers, not text. Tokenization is the bridge from human-readable text to numeric IDs.

```
"I love Python" → ["I", " love", " Python"] → [40, 1842, 11361]
```

### Types of Tokenization

**Word-Level**
```
Input:  "I love machine learning"
Tokens: ["I", "love", "machine", "learning"]
```
Problem: Vocabulary explodes with every word form (run, running, runner, runnable).

**Subword Tokenization (used in modern LLMs)**
```
Input:  "unbelievable"
Tokens: ["un", "believ", "able"]
```
The sweet spot — handles unknown words by splitting into known subwords.

**Character-Level**
```
Input:  "AI"
Tokens: ["A", "I"]
```
Extremely granular — sequences become very long.

### Common Tokenizers

| Model | Tokenizer | Vocabulary Size |
|---|---|---|
| GPT-4 | BPE (Byte Pair Encoding) | ~100,000+ |
| GPT-2 | BPE | ~50,000 |
| BERT | WordPiece | ~30,000 |
| Llama 2/3 | SentencePiece | ~32,000 |
| Mistral | SentencePiece | ~32,000 |

**Practical insight:** Tokenization has real cost implications. More tokens = higher API cost + more latency. Code, structured data (JSON, CSV), and non-English languages often tokenize less efficiently than plain English prose.

**Common misconception:** Tokens are not the same as words. "ChatGPTification" might become ["Chat", "GPT", "ification"] — 3 tokens, 1 word.

---

## 4. Vocabulary and Token IDs

**Vocabulary** is the complete set of tokens a model knows. Each token gets a unique integer ID.

```
Token     → Token ID
"I"       → 40
" love"   → 1842
" Python" → 11361
```

**The embedding matrix** is sized: `Vocabulary Size × Embedding Dimension`

```
Vocabulary Size = 50,000
Embedding Dimension = 768
Embedding Matrix = 50,000 × 768 = 38.4M parameters (just for embeddings)
```

**Small vs Large Vocabulary Trade-offs:**

| | Small Vocabulary | Large Vocabulary |
|---|---|---|
| Memory | Less | More |
| Token sequences | Longer | Shorter |
| Handles rare words | By splitting | More directly |
| Embedding matrix | Smaller | Larger |

**Why subwords win:** They balance vocabulary size with the ability to represent any word by decomposing it into known pieces.

---

## 5. Embeddings

**What they are:** Dense numerical vector representations that capture the *meaning* of tokens. Each token is mapped to a vector of floating-point numbers.

```
"Cat"  → [0.21, -0.45, 0.78, 0.12, ...]
"Dog"  → [0.24, -0.41, 0.75, 0.10, ...]
"Car"  → [0.89,  0.12, -0.34, 0.67, ...]
```

**The key property — semantic similarity:** Tokens with similar meanings have embedding vectors that are close to each other in vector space.

```
King  <-> Queen    (similar — close vectors)
Cat   <-> Dog      (similar — close vectors)
Cat   <-> Database (dissimilar — distant vectors)
```

**Famous example:** `king - man + woman ≈ queen` — arithmetic on meaning is possible in embedding space.

### Why Embeddings Exist

Computers need numbers. But a naive approach (assigning ID 1 to "cat", ID 2 to "dog") gives no information about meaning. Embeddings encode *relationships* between concepts as geometric distances.

### Contextual vs Static Embeddings

- **Static embeddings** (Word2Vec, GloVe): Each word has one fixed vector regardless of context. "Bank" gets the same vector in "river bank" and "bank account".
- **Contextual embeddings** (BERT, GPT): The embedding for each token is influenced by surrounding tokens. This is what Transformer models produce internally.

### Embedding Dimension

Larger dimensions capture richer meaning but cost more memory and computation.

| Model | Embedding Dimension |
|---|---|
| GPT-2 (small) | 768 |
| GPT-3 | 12,288 |
| Llama 3 8B | 4,096 |

---

## 6. Positional Encoding

**The problem:** Transformers process all tokens in parallel. Without explicit position information, "Dog bites man" and "Man bites dog" look identical — same tokens, different order.

**The solution:** Add a positional signal to each token's embedding before feeding into the Transformer.

```
Final Token Representation = Token Embedding + Positional Encoding
```

### Types

**Sinusoidal (Original Transformer)**
- Uses mathematical sine/cosine functions to generate position vectors
- No additional parameters to learn
- Can generalize to sequence lengths longer than those seen during training

**Learned Positional Encoding**
- The model learns position vectors during training
- Typically better performance in practice
- Used by GPT-2 and many early models

**Rotary Position Embedding (RoPE)**
- Used by Llama, Mistral, and many modern models
- Encodes relative positions, not absolute ones
- Better for long contexts and generalizes well

**Why it matters:** Without positional encoding, a Transformer is a bag-of-tokens model — word order is meaningless. Positional encoding is what gives the model awareness of sequence structure.

---

## 7. Self-Attention and Q, K, V

**Self-Attention is the core innovation of Transformers.** It allows each token to look at every other token in the sequence and decide how much attention to pay to each one.

### The Intuition

Sentence: *"The animal didn't cross the road because it was tired."*

To understand what "it" refers to, a model must connect "it" back to "animal". Self-Attention does this by computing a relationship score between every pair of tokens.

### The Library Analogy

Think of searching for a book in a library:
- **Query (Q):** What you're looking for ("books about Python web frameworks")
- **Key (K):** Labels/metadata on every book ("FastAPI", "Django", "Flask")
- **Value (V):** The actual content of each book

```
Query
  ↓
Compare with all Keys
  ↓
Find best-matching Keys
  ↓
Retrieve corresponding Values
  ↓
Weighted combination → output
```

### How Q, K, V Are Created

For each input token embedding, three separate linear projections create Q, K, V vectors:

```
Token Embedding
      ↓
  x W_Q  → Q (Query)
  x W_K  → K (Key)
  x W_V  → V (Value)
```

The weight matrices W_Q, W_K, W_V are learned during training.

### Attention Score Calculation (High Level)

```
Attention Score = softmax( Q × K^T / sqrt(d_k) ) × V
```

Where `d_k` is the dimension of the key vectors (scaling prevents extremely small gradients).

### Example

Sentence: *"I love AI"* — computing attention for the token "AI":

| Token | Attention Score | Meaning |
|---|---|---|
| I | 0.1 | Low relevance |
| love | 0.3 | Medium relevance |
| AI | 0.6 | High relevance (self-reference) |

The output for "AI" is a weighted combination of all Value vectors — heavily weighted toward the most relevant tokens.

**Interview insight:** Self-Attention has O(n²) complexity with respect to sequence length. This is why long contexts are expensive — doubling the context length quadruples the attention computation.

---

## 8. Multi-Head Attention

**The problem with single-head attention:** One attention head can only look at the sentence from one perspective at a time.

**The solution:** Run multiple attention heads in parallel, each learning to capture different types of relationships, then combine their outputs.

```
Input
  ↓
Q, K, V
  ↓
[Head 1: Grammar] [Head 2: Coreference] [Head 3: Semantics] ...
         ↓
    Concatenate
         ↓
    Linear Layer
         ↓
       Output
```

### Example: What Different Heads Capture

Sentence: *"The cat sat on the mat because it was tired."*

| Head | Focus |
|---|---|
| Head 1 | it → cat (coreference) |
| Head 2 | sat → mat (spatial relationship) |
| Head 3 | cat → tired (state attribution) |
| Head 4 | sentence-level grammatical structure |

**Analogy:** Imagine reviewing a film. One person focuses on the story, another on the acting, another on cinematography. Multi-Head Attention is the model's ability to analyze language from multiple critical perspectives simultaneously.

---

## 9. The Transformer Block

The Transformer Block is the repeating unit stacked N times to form the full model. Modern LLMs have anywhere from 12 to 96+ blocks.

```
Input
  ↓
Multi-Head Attention
  ↓
Add & Normalize (Residual Connection + Layer Norm)
  ↓
Feed-Forward Network (FFN)
  ↓
Add & Normalize
  ↓
Output (passed to the next block)
```

### Components in Detail

**Residual Connection (Add):**
```
Output = SubLayer(Input) + Input
```
Why? Deep networks suffer from vanishing gradients. Adding the original input gives gradients a "shortcut" to flow backward during training, enabling much deeper models.

**Layer Normalization:** Normalizes activations to have zero mean and unit variance. Stabilizes training and speeds up convergence.

**Feed-Forward Network (FFN):** A two-layer neural network applied independently to each token position.
```
Input → Linear → GELU/ReLU → Linear → Output
```
The FFN typically expands the dimension by 4x internally (e.g., 768 → 3072 → 768). Believed to store factual knowledge.

### Progressive Understanding Across Blocks

```
Block 1  → Basic word relationships, syntax
Block 2  → Phrase-level context
Block 3  → Sentence meaning
...
Block N  → Abstract reasoning, complex semantics
```

**Scale:** GPT-3 has 96 Transformer blocks. Llama 3 8B has 32 blocks.

---

## 10. Linear Layer + Softmax

After all N Transformer blocks, two final layers convert the rich token representation into a probability distribution over the vocabulary.

```
Final Hidden State
       ↓
  Linear Layer
       ↓
    Logits (one score per vocabulary token)
       ↓
    Softmax
       ↓
  Probabilities (sum = 1.0)
       ↓
  Next Token Selection
```

### Logits and Softmax

**Logits** are raw scores — can be any real number, don't sum to 1.
```
AI      → 8.5
Python  → 6.3
ML      → 2.1
Data    → 1.4
```

**Softmax** converts to probabilities:
```
AI      → 88%
Python  → 11%
ML      → 0.8%
Data    → 0.2%
```

### Token Selection Strategies

- **Greedy Decoding:** Always pick the highest probability token. Deterministic but repetitive.
- **Temperature Sampling:** Scale logits before softmax. High temperature → more random. Low temperature → more deterministic.
- **Top-K Sampling:** Only sample from the K highest-probability tokens.
- **Top-P (Nucleus) Sampling:** Sample from the smallest set whose cumulative probability exceeds P.

---

## 11. Detokenization

**Detokenization** converts the generated sequence of token IDs back into human-readable text — the reverse of tokenization.

```
Token IDs: [40, 1842, 11361]
Output:    "I love Python"
```

For subword tokens, pieces are joined and spacing/punctuation is restored per the tokenizer's rules. The tokenizer used for decoding must match the one used for encoding.

---

## 12. The Training Phase

**Training** adjusts the model's weights so it gets better at predicting the next token. Everything the model "knows" is encoded in its weights.

### The Learning Loop

```
Input:  "I love learning"
Target: "AI"
    ↓
1. Forward Pass → Prediction: "ML" (wrong)
2. Loss Calculation (cross-entropy)
3. Backpropagation → which weights caused the error?
4. Gradient Descent → update those weights slightly
5. Repeat billions of times across trillions of tokens
```

### Scale of Training

| Model | Training Tokens | 
|---|---|
| GPT-3 | 300B |
| Llama 3 8B | 15T |
| Llama 3 70B | 15T |

### Training vs Inference

| | Training | Inference |
|---|---|---|
| Weights | Being updated | Fixed |
| Gradients | Computed | Not computed |
| Memory | Much higher | Lower |
| Speed | Slow | Fast |
| When | Once (or occasionally) | Every user request |

---

## 13. Self-Supervised Learning

**The problem:** Supervised learning requires labeled data. At the scale needed for LLMs (trillions of tokens), manual labeling is impossible.

**Self-Supervised Learning** generates labels automatically from the data itself.

```
Text: "The capital of France is Paris"

Auto-generated training example:
  Input:  "The capital of France is ___"
  Label:  "Paris"
```

No human labeled this. The answer exists in the text.

### Two Approaches

**GPT-style (Causal / Next Token Prediction):**
Predict the next token given all previous tokens. Enables text generation.

**BERT-style (Masked Language Modeling):**
Randomly mask tokens and predict them. Enables bidirectional understanding.

| | GPT-style | BERT-style |
|---|---|---|
| Predicts | Next token | Masked tokens |
| Context | Left only | Both directions |
| Good for | Text generation | Understanding, classification |

---

## 14. What "Pre-Trained" Means

**Pre-training** is the large-scale first phase where the model learns general language understanding from massive, diverse datasets using self-supervised learning.

```
Massive Text Data (books, web, code, papers)
           ↓
Self-Supervised Training
           ↓
Pre-Trained Model
   → Understands language
   → Knows facts
   → Reasons at a basic level
   → Knows coding patterns
```

**GPT = Generative Pre-trained Transformer**

### What Happens After Pre-Training

```
Pre-Trained Base Model
        ↓
Instruction Fine-Tuning   ← Teach it to follow instructions
        ↓
RLHF / DPO / Constitutional AI  ← Align to human preferences
        ↓
Chat/Assistant Model
```

**Key insight:** The base model alone just continues text — it is not an assistant. The alignment phase is what creates ChatGPT, Claude, Gemini, etc.

---

## 15. Encoder vs Decoder

Transformers come in three architectural variants. The variant determines what the model is best suited for.

### Encoder-Only

```
Input Text → Encoder → Contextual Representation
```

- Bidirectional attention — every token sees all other tokens
- Cannot generate new text on its own
- Examples: **BERT**, RoBERTa, DistilBERT
- Best for: Classification, sentiment analysis, NER, semantic embeddings for RAG

### Decoder-Only

```
Input Prompt → Decoder → Generated Token → ... → Text
```

- Causal (masked) attention — each token sees only previous tokens
- Generates text auto-regressively
- Examples: **GPT**, **Llama**, **Mistral**, **Claude**
- Best for: Chat, code generation, all generative tasks

### Encoder-Decoder

```
Input → Encoder → Context → Decoder → Output
```

- Encoder understands input; Decoder generates output conditioned on it
- Examples: **T5**, BART
- Best for: Translation, summarization

### Attention Types

| Architecture | Attention | Sees future tokens? |
|---|---|---|
| Encoder-Only | Bidirectional | Yes |
| Decoder-Only | Causal (Masked) | No |
| Encoder-Decoder | Both + Cross-Attention | Encoder: Yes, Decoder: No |

**Why most modern LLMs are decoder-only:** Pre-training on next-token prediction naturally produces a decoder. Decoder-only models also scale exceptionally well.

---

## 16. Fine-Tuning

Fine-Tuning continues training a pre-trained model on a smaller, task-specific dataset to adapt its behavior.

```
Pre-Trained Model (general)
          ↓
Task-Specific Dataset
          ↓
Fine-Tuning
          ↓
Specialized Model
```

### Types

**Full Fine-Tuning:** All parameters updated. Highest quality, highest cost.

**LoRA (Low-Rank Adaptation):**
```
Original Weight W (frozen)
+ Small trainable matrices A × B  (rank << original dimensions)
Effective weight = W + A × B
```
Only the small adapter matrices are trained — typically <1% of total parameters. After training, can be merged back or kept separate.

**QLoRA:** LoRA on a quantized base model. Enables fine-tuning large models on consumer hardware.

### When to Fine-Tune vs Use RAG

| | Fine-Tuning | RAG |
|---|---|---|
| Updates model weights | Yes | No |
| Best for | Behavioral changes, style, format | Knowledge retrieval |
| Data freshness | Requires retraining | Easy to update |
| Cost | Higher upfront | Lower upfront |

**Practical rule:** Use RAG for knowledge, fine-tune for behavior.

---

## 17. Inference and Sampling Strategies

**Inference** is running the trained model to generate output. Weights are frozen.

### Auto-Regressive Generation

LLMs generate one token at a time, each conditioned on all previous tokens:

```
Input: "The capital of France is"
→ "Paris"  → "."  → "<EOS>"
Output: "Paris."
```

### Sampling Controls

**Temperature:** Controls randomness.
```
Low temp (0.1–0.5): More deterministic → AI=98%, Python=1.9%, ML=0.1%
High temp (1.2–2.0): More random → AI=45%, Python=35%, ML=20%
```

**Top-K:** Sample only from top K tokens by probability.

**Top-P (Nucleus):** Sample from smallest set with cumulative probability ≥ P.

### Inference Optimization

**KV Cache:** Stores Key and Value matrices from previous tokens — avoids recomputation on each step. Critical for performance.

**Speculative Decoding:** A small fast model generates candidates; the large model verifies multiple at once.

---

## 18. Context Window

**Context Window** is the maximum number of tokens an LLM can process at once (input + output combined).

| Model | Context Window |
|---|---|
| GPT-3.5 | 4K – 16K |
| GPT-4o | 128K |
| Claude 3.5 Sonnet | 200K |
| Llama 3.1 405B | 128K |
| Gemini 1.5 Pro | 1M |

**Why limited:** Self-Attention is O(n²) — doubling context length quadruples computation.

**Rule of thumb:** ~750 words ≈ 1,000 tokens

**Engineering implication:** RAG systems retrieve only the most relevant chunks to stay within the context window while providing useful information.

---

## 19. Hallucinations

**Hallucinations** are confident, fluent, but factually incorrect outputs. The model says something that sounds right but isn't.

### Why They Happen

LLMs predict the most *plausible* next token — not the most *accurate* one. They have no mechanism to verify facts against ground truth.

### Types

- **Factual:** Wrong facts stated confidently
- **Temporal:** Outdated knowledge as current
- **Source:** Fabricated citations
- **Reasoning:** Correct-sounding but wrong logic

### Mitigation

| Strategy | How It Helps |
|---|---|
| RAG | Ground answers in retrieved facts |
| Temperature = 0 | More deterministic |
| Prompt engineering | Instruct to say "I don't know" |
| Fine-tuning | Train on accurate domain data |
| Chain of Thought | Forces explicit reasoning |

**Interview insight:** Hallucinations are a fundamental property of the architecture, not a simple bug. They arise from the training objective — plausibility, not accuracy.

---

## 20. Chain of Thought Reasoning

**Chain of Thought (CoT)** encourages the model to reason through a problem step-by-step before giving the final answer.

### Without vs With CoT

**Without:**
```
Q: Roger has 5 apples. Buys 3 more. Gives away 2. How many?
A: 6
```

**With (append "Let's think step by step"):**
```
Roger starts with 5.
He buys 3 more: 5 + 3 = 8.
He gives away 2: 8 - 2 = 6.
Answer: 6
```

### Why It Works

Intermediate reasoning steps occupy the model's context and act as working memory. Each step conditions the next, breaking complex multi-step problems into simpler single-step predictions.

### Types

- **Zero-Shot CoT:** Append "Let's think step by step."
- **Few-Shot CoT:** Provide worked examples with visible reasoning chains.

---

## 21. Reasoning Models

**Reasoning Models** are LLMs specifically trained to perform extended deliberate reasoning before answering.

```
User Question
      ↓
[Extended Internal Reasoning — many tokens]
  → Consider approach A...
  → That fails because...
  → Try approach B...
  → Verify answer...
      ↓
Final Answer
```

### Standard LLM vs Reasoning Model

| Feature | Standard LLM | Reasoning Model |
|---|---|---|
| Response | Direct answer | Reasoning trace + answer |
| Complex math | Moderate | Strong |
| Latency | Lower | Higher |

### Examples

| Model | Organization |
|---|---|
| o1, o3, o4-mini | OpenAI |
| Claude (extended thinking) | Anthropic |
| DeepSeek-R1 | DeepSeek |
| Gemini 2.0 Flash Thinking | Google |

**CoT vs Reasoning Models:** CoT is a prompting technique. Reasoning capability is baked into reasoning models through training.

---

## 22. Small Language Models

**SLMs** are models with fewer parameters designed for efficiency, local deployment, and specialized tasks.

| Feature | SLM | LLM |
|---|---|---|
| Parameters | Millions–few billion | Tens–hundreds of billions |
| Speed | Faster | Slower |
| Can run locally | Yes | Usually No |
| General knowledge | Limited | Extensive |

### Popular SLMs

| Model | Size | Organization |
|---|---|---|
| Phi-4 Mini | 3.8B | Microsoft |
| Gemma 3 2B/4B | 2B–4B | Google |
| Llama 3.2 3B | 3B | Meta |
| Mistral 7B | 7B | Mistral AI |
| TinyLlama | 1.1B | Open Source |

---

## 23. Knowledge Distillation

**Knowledge Distillation** trains a small **student model** to mimic a large **teacher model** by learning from the teacher's soft probability distributions (not just correct/incorrect labels).

```
Teacher's soft labels: Paris=95%, Lyon=3%, Nice=2%
Student learns richer signal than just: Paris=correct
```

### Workflow

```
Large Teacher Model
        ↓
Generate soft probability distributions
        ↓
Student trains to match distributions
        ↓
Efficient, smaller student model
```

---

## 24. Quantization

**Quantization** reduces model size by storing weights in lower numerical precision.

```
FP32: 3.14159265  (32 bits)
INT8: 3.14        (8 bits, ~4x smaller)
INT4: 3.1         (4 bits, ~8x smaller)
```

### Why It Matters

Llama 3 70B at FP32 requires ~280GB GPU memory. At INT4, ~35GB — runnable on a high-end workstation.

### Tools

| Tool | Use Case |
|---|---|
| bitsandbytes | 8-bit/4-bit in Python |
| GGUF / llama.cpp | Run quantized models locally |
| GPTQ | Post-training quantization |
| AWQ | Activation-aware (better quality) |
| TensorRT | NVIDIA production optimization |

---

## 25. Special Tokens

Special tokens are control signals embedded in the vocabulary.

| Token | Meaning |
|---|---|
| `<BOS>` / `<s>` | Beginning of sequence |
| `<EOS>` / `</s>` | End of sequence — stop generating |
| `<PAD>` | Padding for batch processing |
| `<UNK>` | Unknown token |
| `<MASK>` | Masked token (BERT-style) |
| `[INST]` | Instruction delimiters (Llama) |

**Why this matters:** Each model family has its own chat template using different special tokens. Using the wrong template silently degrades output quality.

---

## Complete Mental Model

```
Pre-training on trillions of tokens
             ↓
Text → Tokenization → Token IDs
             ↓
Token IDs → Embeddings + Positional Encoding
             ↓
[Transformer Block × N]
  ├── Multi-Head Self-Attention (Q, K, V)
  ├── Add & Normalize
  ├── Feed-Forward Network
  └── Add & Normalize
             ↓
Linear Layer → Logits → Softmax → Probabilities
             ↓
Next Token (via temperature/top-p/top-k)
             ↓
Detokenization → Human-Readable Text
             ↓
            LLM

To get a useful assistant:
Base Model → SFT → RLHF/DPO → Chat Model

To deploy efficiently:
Full precision → Quantization → KV Cache + Batch Inference → Production API
```
