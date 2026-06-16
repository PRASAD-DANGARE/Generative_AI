# Tokens, Context Window & Costing
### A Complete Reference for Backend Engineers Transitioning to AI Engineering

> **How to use this document:** Every concept answers — *What is it? Why does it exist? How does it work? How does it connect? What are the trade-offs?* Read Section A first (Tokens & Tokenization) — every other section builds on it.

---

## Table of Contents

**A. Tokens & Tokenization**
1. [What is a Token?](#1-what-is-a-token)
2. [What is Tokenization?](#2-what-is-tokenization)
3. [Why do LLMs need Tokenization?](#3-why-do-llms-need-tokenization)
4. [Why are Tokens the Basic Unit of Processing?](#4-why-are-tokens-the-basic-unit)
5. [Why don't LLMs process whole sentences?](#5-why-not-whole-sentences)
6. [Why don't LLMs process whole words?](#6-why-not-whole-words)
7. [Why can't we use character-level processing?](#7-why-not-characters)
8. [Characters vs Tokens vs Words](#8-characters-vs-tokens-vs-words)
9. [Benefits of Tokenization](#9-benefits-of-tokenization)
10. [Types of Tokenization](#10-types-of-tokenization)
11. [Word-Level Tokenization](#11-word-level-tokenization)
12. [Character-Level Tokenization](#12-character-level-tokenization)
13. [Subword Tokenization](#13-subword-tokenization)
14. [Common Tokenization Algorithms](#14-tokenization-algorithms)
15. [Vocabulary](#15-vocabulary)
16. [Vocabulary Size](#16-vocabulary-size)
17. [Special Tokens](#17-special-tokens)
18. [Token IDs](#18-token-ids)
19. [Token IDs vs Tokens](#19-token-ids-vs-tokens)
20. [Tokenization Pipeline](#20-tokenization-pipeline)
21. [What happens after Tokenization?](#21-what-happens-after-tokenization)
22. [Common Misconceptions about Tokens](#22-common-misconceptions)

**B. Token Count**

23. [What is Token Count?](#23-what-is-token-count)
24. [Why does Token Count matter?](#24-why-does-token-count-matter)
25–30. [How Token Count affects Cost, Latency, Context, Scalability, Quality](#25-token-count-effects)
31. [Why AI Engineers care about Token Count](#31-why-ai-engineers-care)

**C. Context Window**

32. [What is a Context Window?](#32-what-is-a-context-window)
33. [Why do LLMs need a Context Window?](#33-why-llms-need-context-window)
34. [What contributes to the Context Window?](#34-what-contributes-to-context)
35. [Input Tokens vs Output Tokens within Context](#35-input-vs-output-in-context)
36. [Context Window Sizes of Popular Models](#36-context-window-sizes)
37. [Why is the Context Window Limited?](#37-why-context-window-limited)
38. [Self-Attention Complexity and Context Limits](#38-self-attention-complexity)
39. [Advantages of Large Context Windows](#39-large-context-advantages)
40. [Disadvantages of Large Context Windows](#40-large-context-disadvantages)
41. [Advantages of Small Context Windows](#41-small-context-advantages)
42. [Disadvantages of Small Context Windows](#42-small-context-disadvantages)
43. [Context Window vs Memory](#43-context-window-vs-memory)
44. [Context Window vs Long-Term Memory](#44-context-window-vs-long-term-memory)
45. [Context Overflow](#45-context-overflow)
46. [Context Management Strategies](#46-context-management-strategies)
47. [Context Compression](#47-context-compression)
48. [Context Summarization](#48-context-summarization)
49. [Sliding Window Approach](#49-sliding-window-approach)

**D. Input & Output Tokens**

50. [What are Input Tokens?](#50-what-are-input-tokens)
51. [What are Output Tokens?](#51-what-are-output-tokens)
52. [Input Tokens vs Output Tokens](#52-input-vs-output-tokens)
53. [What counts as Input Tokens?](#53-what-counts-as-input-tokens)
54. [What counts as Output Tokens?](#54-what-counts-as-output-tokens)
55. [Why are Output Tokens usually more expensive?](#55-why-output-tokens-cost-more)
56. [How Output Tokens are generated sequentially](#56-sequential-generation)
57. [Input vs Output Trade-offs](#57-input-output-tradeoffs)

**E. API Costing**

58. [How API Pricing Works](#58-api-pricing)
59. [Pricing based on Token Usage](#59-pricing-per-token)
60. [Input Pricing vs Output Pricing](#60-input-vs-output-pricing)
61. [API Cost Formula](#61-api-cost-formula)
62. [Factors affecting API Cost](#62-cost-factors)
63. [Cost Estimation in Production](#63-cost-estimation)
64. [Token Usage Monitoring](#64-token-monitoring)
65. [Cost Optimization Techniques](#65-cost-optimization)
66. [Cached Tokens](#66-cached-tokens)
67. [Batch Processing](#67-batch-processing)
68. [Model Selection and Cost Trade-offs](#68-model-selection)

**F. Long Context Challenges**

69. [The "Lost in the Middle" Problem](#69-lost-in-the-middle)
70. [Why "Lost in the Middle" happens](#70-why-lost-in-middle)
71. [Impact on RAG](#71-impact-on-rag)
72. [How to mitigate "Lost in the Middle"](#72-mitigate-lost-in-middle)
73. [Why more context doesn't always improve answers](#73-more-context-not-always-better)
74. [Quality vs Quantity of Context](#74-quality-vs-quantity)
75. [Chunk Ordering and Reranking](#75-chunk-ordering)

**G. Production & Interview Perspectives**

76. [Backend Engineer Mental Models](#76-backend-mental-models)
77. [Production Considerations](#77-production-considerations)
78. [Common Misconceptions](#78-common-misconceptions)
79. [Interview Questions & Answers](#79-interview-questions)

---

# A. Tokens & Tokenization

---

## 1. What is a Token?

A **token** is the smallest unit of text that an LLM works with. It is not always a word, not always a character — it is a chunk of text that the model's vocabulary recognises as a single unit.

Think of tokens as the "atoms" of text for an LLM. Just as you cannot split a hydrogen atom and still call it hydrogen, an LLM cannot split a token and still process it meaningfully — a token is the minimum indivisible unit of input.

**Examples:**

```
"Hello"         → 1 token   ["Hello"]
"Hello world"   → 2 tokens  ["Hello", " world"]
"ChatGPT"       → 3 tokens  ["Chat", "G", "PT"]    ← subword splits
"unbelievable"  → 3 tokens  ["un", "believ", "able"]
"2024"          → 1 token   ["2024"]
"!"             → 1 token   ["!"]
```

**Key fact:** On average, 1 token ≈ 4 characters in English, or roughly 0.75 words. So 1,000 tokens ≈ 750 words ≈ 1.5 pages of text.

**Why this matters for you as an engineer:** Every API call costs money per token. Every model has a token limit. Understanding what a token is determines how you design prompts, estimate costs, and manage context.

---

## 2. What is Tokenization?

**Tokenization** is the process of converting raw text into a sequence of tokens (and then into token IDs) that an LLM can process.

It is the very first step in the LLM pipeline — raw text cannot enter the model directly. Tokenization is the translation layer between human language and model mathematics.

**Full pipeline:**

```
Raw Text
    ↓
Tokenization
    ↓
Token Sequence    ["I", " love", " Python"]
    ↓
Token IDs         [40, 1842, 11361]
    ↓
Embeddings        [[0.21, -0.45, ...], [0.67, 0.12, ...], ...]
    ↓
LLM processes numbers, not text
```

**Backend analogy:** Tokenization is like serialisation. Before you send an object over the network, you serialise it to JSON or Protobuf — a format the receiver understands. Tokenization serialises text into a format the LLM understands (numbers).

---

## 3. Why do LLMs need Tokenization?

LLMs are mathematical functions. They operate entirely on numbers — specifically, vectors of floating-point numbers. They cannot read text directly.

**The problem:** Text is discrete, variable-length, and symbolic. Neural networks need fixed-size numerical inputs.

**Tokenization solves this by:**
- Converting text → integers (token IDs)
- Mapping integers → dense vectors (embeddings)
- Giving the model a fixed-size vocabulary to reason over

**Without tokenization, there is no way to feed text into a neural network.** Every NLP system — not just LLMs — needs some form of tokenization.

**The three things tokenization enables:**

| Need | How Tokenization Helps |
|---|---|
| Numbers, not text | Converts text to integer IDs |
| Fixed vocabulary | Every possible input maps to known tokens |
| Manageable sequences | Subword tokens keep sequences from being too long |

---

## 4. Why are Tokens the Basic Unit of Processing?

The LLM predicts **one token at a time**. Not one character, not one word, not one sentence — one token.

At every step during generation, the model looks at all tokens so far and predicts the probability distribution over the entire vocabulary for the next token. It then picks one token from that distribution.

This is the fundamental loop:

```
[Token1, Token2, ..., TokenN] → predict TokenN+1
[Token1, Token2, ..., TokenN, TokenN+1] → predict TokenN+2
...repeat until EOS token
```

The token is the basic unit because:
- The vocabulary (all known units) is defined in terms of tokens
- The embedding matrix maps token IDs to vectors — one row per token
- The softmax output layer produces one probability per token in the vocabulary
- Attention is computed across token positions

Everything in the model — from input to output — is built around the token as the atomic unit.

---

## 5. Why don't LLMs process whole sentences?

If the model processed whole sentences as single units, the vocabulary would need to contain every possible sentence — which is infinite. You cannot build a finite embedding matrix for an infinite vocabulary.

Additionally, sentences vary enormously in length. You cannot feed variable-length symbolic units into a fixed mathematical function without a standard atomic unit to count and represent.

Tokens provide that fixed atomic unit. Sentences are just sequences of tokens — the model sees structure at the token level and learns sentence-level patterns from the positions and relationships between tokens.

---

## 6. Why don't LLMs process whole words?

Whole-word processing was tried early in NLP (word-level tokenization). It has two critical problems:

**Problem 1 — Vocabulary explosion:**
English has hundreds of thousands of words. Add other languages, technical terms, code identifiers, and misspellings — you need millions of vocabulary entries. Each entry requires a row in the embedding matrix. This gets enormous very quickly.

**Problem 2 — Out-of-vocabulary (OOV) words:**
Any word the model hasn't seen during training cannot be represented. "ChatGPTification" is not in any pre-trained word vocabulary. The model has no way to handle it.

**Subword tokens solve both problems:** the vocabulary stays manageable (30K–100K entries) and any word — even made-up ones — can be represented by combining known subword pieces.

---

## 7. Why can't we use character-level processing?

Character-level processing is technically possible but has serious practical problems.

**The core problem — sequence length:**

Consider: "I love building AI systems" — 26 characters = 26 tokens at character level, but only 6 tokens at subword level.

Every downstream operation in an LLM — especially self-attention — scales with sequence length. Character-level sequences are 4–10x longer than subword sequences for the same text. This means:

- Training is 4–10x more expensive
- Inference is 4–10x slower
- The context window fits 4–10x less content
- Self-attention (which is O(n²)) gets 16–100x more expensive computationally

**The other problem — learning difficulty:**
The model has to learn that "c", "a", "t" together means cat — a much harder task than starting with "cat" as an atomic unit. Subword models start with more structure, making learning faster and more efficient.

Character-level models are used in some specialised settings (character-aware language models for morphologically rich languages) but are not practical for general-purpose LLMs.

---

## 8. Characters vs Tokens vs Words

This comparison is one of the most commonly confused topics in LLM engineering.

| Unit | Example | Count for "unbelievable" | Pros | Cons |
|---|---|---|---|---|
| Character | "u","n","b","e","l","i","e","v","a","b","l","e" | 12 | No OOV ever | Very long sequences, slow |
| Word | "unbelievable" | 1 | Short sequences | OOV problem, huge vocabulary |
| Subword token | "un","believ","able" | 3 | Balance of both | Requires a trained tokenizer |

**Concrete comparison for the sentence "I love learning Python":**

```
Character-level: ["I"," ","l","o","v","e"," ","l","e","a","r","n","i","n","g"," ","P","y","t","h","o","n"] = 22 tokens
Word-level:      ["I","love","learning","Python"] = 4 tokens
Subword (GPT-2): ["I"," love"," learning"," Python"] = 4 tokens
```

For simple common words, subword and word-level are similar. The difference shows with rare or complex words:

```
"ChatGPTification"
Character-level: 17 characters = 17 tokens
Word-level:      1 token (but likely OOV — never seen before!)
Subword:         ["Chat","G","PT","ification"] = 4 tokens (handles it cleanly)
```

**The rule of thumb for English text:**
- 1 token ≈ 4 characters
- 1 token ≈ 0.75 words
- 1,000 tokens ≈ 750 words ≈ 3/4 of a page

---

## 9. Benefits of Tokenization

**1. Bridges text and mathematics**
Makes it possible to feed language into a neural network at all.

**2. Handles any input**
Subword tokenization can represent any word — even ones the model never saw during training — by breaking them into known pieces.

**3. Manageable vocabulary**
30K–100K tokens instead of millions of words or billions of character combinations.

**4. Efficient sequences**
Subword tokens produce shorter sequences than character-level, making the model faster and cheaper.

**5. Language agnosticism**
Modern tokenizers (especially BPE on byte level) can handle any language, code, mathematical symbols, emoji, and binary data without special handling.

**6. Consistent representation**
The same tokenizer is used at training time and inference time, guaranteeing the model always sees the same representation for the same text.

---

## 10. Types of Tokenization

There are three major approaches, each making a different trade-off between vocabulary size and sequence length:

```
                    Vocabulary Size
                    Small ←————————————→ Large
                    
Sequence Length
Long ↑      Character-level
             (vocab: ~256, sequences: very long)

             Subword-level       ← sweet spot
             (vocab: 30K-100K, sequences: moderate)
Short ↓                                    Word-level
                                           (vocab: millions, sequences: short)
```

The subword approach wins in practice because it sits at the optimal point: small enough vocabulary to be tractable, short enough sequences to be efficient.

---

## 11. Word-Level Tokenization

**How it works:** Split text on whitespace and punctuation. Each unique word is a vocabulary entry.

```
Input:  "I love machine learning"
Tokens: ["I", "love", "machine", "learning"]
IDs:    [42, 1823, 7341, 5629]
```

**Pros:**
- Simple to implement
- Short sequences (one token per word)
- Human-readable tokens

**Cons:**
- Massive vocabulary (hundreds of thousands of entries)
- Cannot handle unseen words (OOV → mapped to `<UNK>`)
- Treats "run", "running", "runner" as completely unrelated words
- Fails badly on code, URLs, technical terms, and non-English text

**Where it was used:** Early NLP systems (pre-2018). Not used in modern LLMs.

---

## 12. Character-Level Tokenization

**How it works:** Every individual character is a token. Vocabulary is tiny (256 ASCII chars + Unicode extensions).

```
Input:  "cat"
Tokens: ["c", "a", "t"]
IDs:    [99, 97, 116]
```

**Pros:**
- Zero OOV problem — every possible input is representable
- Tiny vocabulary (hundreds of entries vs thousands)
- Handles any language, code, emoji, symbols natively

**Cons:**
- Sequences are 4–10x longer → expensive attention, smaller effective context
- Model must learn that "c"+"a"+"t" means something — harder than starting with "cat"
- Training and inference are much slower for the same text content

**Where it is used:** Character-aware models for morphologically rich languages (Finnish, Turkish), certain specialised NLP tasks, and as a fallback within byte-level BPE for rare characters.

---

## 13. Subword Tokenization

**How it works:** Frequently occurring character sequences become their own tokens. Rare or unseen words are split into recognisable pieces.

```
Input:  "unbelievable"
Tokens: ["un", "believ", "able"]

Input:  "tokenization"
Tokens: ["token", "ization"]

Input:  "ChatGPTification"
Tokens: ["Chat", "G", "PT", "ification"]
```

**The key insight:** Common words stay as one token. Rare or novel words are broken into subword pieces. The model learns that "un-" is a negation prefix by seeing it appear with many different stems — even though it never saw "unbelievable" as a whole word, it understands the parts.

**Why this is the sweet spot:**
- Vocabulary stays small and manageable (32K–100K tokens)
- No OOV problem — anything can be decomposed into subwords
- Shorter sequences than character-level
- Model can generalise across word forms via shared subwords

**Used by:** Every major modern LLM — GPT-4, Llama, Claude, Mistral, BERT, T5, and more.

---

## 14. Common Tokenization Algorithms

### BPE — Byte Pair Encoding

**Origin:** GPT-2, GPT-3, GPT-4, RoBERTa

**How it works:**
1. Start with individual characters (or bytes) as the initial vocabulary
2. Count all pairs of adjacent tokens in the training corpus
3. Merge the most frequent pair into a new single token
4. Repeat until vocabulary reaches the target size (e.g., 50,000)

```
Training data: "aaabdaaabac"

Step 1: Count pairs → "aa" appears 4 times (most frequent)
Step 2: Merge "aa" → "Z": "ZabdZabac"
Step 3: Count pairs → "Za" appears 2 times
Step 4: Merge "Za" → "Y": "YbdYbac"
...continue until vocabulary size reached
```

**Result:** Frequent character combinations become single tokens. "ing", "tion", "pre", "un" all become single tokens because they appear millions of times in training data.

**Byte-level BPE (GPT-4):** Operates on raw bytes instead of characters, allowing it to handle any Unicode text without special cases.

**Vocabulary size:** GPT-2 = 50,257 | GPT-4 = ~100,000+

---

### WordPiece

**Origin:** BERT, DistilBERT

**How it works:** Similar to BPE but instead of merging the most frequent pair, it merges the pair that maximises the likelihood of the training data. Uses `##` prefix to mark tokens that continue a word (not word-start).

```
"unbelievable" → ["un", "##believ", "##able"]
```

The `##` prefix tells the model "this token is a continuation, not the start of a new word".

**Vocabulary size:** BERT = 30,522

---

### SentencePiece

**Origin:** Llama, Mistral, T5, many multilingual models

**How it works:** Treats the input as a raw sequence of Unicode characters with no pre-tokenization on whitespace. This makes it truly language-agnostic — it handles Japanese, Arabic, and Chinese (which have no spaces) as naturally as English.

```
"I love Python"
WordPiece → ["I", "love", "Python"]  (splits on spaces first)
SentencePiece → ["▁I", "▁love", "▁Python"]  (▁ marks word boundaries)
```

The `▁` (underscore) marks the start of a word in the raw byte stream, allowing reconstruction of the original spacing during detokenization.

**Vocabulary size:** Llama 2 = 32,000 | Llama 3 = 128,256 (much larger)

---

### Tiktoken (OpenAI)

OpenAI's custom BPE tokenizer used in GPT-3.5, GPT-4, and the embedding models. It is byte-level BPE, very fast, and handles all Unicode natively.

You can use it directly in Python:
```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")
tokens = enc.encode("Hello, how are you?")
print(tokens)        # [9906, 11, 1268, 527, 499, 30]
print(len(tokens))   # 6
```

---

## 15. Vocabulary

The **vocabulary** is the complete, fixed set of all tokens a model knows. It is determined during tokenizer training and never changes after that.

Think of vocabulary as the model's "alphabet" — but instead of 26 letters, it has 30,000–100,000 entries.

**What vocabulary contains:**
- Common words: "the", "and", "Python", "function"
- Subword pieces: "ing", "tion", "un", "##able"
- Individual characters: "a", "b", "c" (as fallbacks)
- Special tokens: `<BOS>`, `<EOS>`, `<PAD>`, `<UNK>`
- Numbers and symbols: "2024", "!", "→", "@"
- Code tokens: "def", "import", "return", "::"

**Backend analogy:** The vocabulary is like an enum or a lookup table. Every possible token has a unique integer ID. Any text you can ever write maps to a sequence of these IDs.

---

## 16. Vocabulary Size

**Vocabulary size** is the total number of unique tokens in the vocabulary. This is a crucial architectural decision — it affects model size, training efficiency, and how well the model handles different languages and domains.

| Model | Tokenizer | Vocabulary Size |
|---|---|---|
| GPT-2 | BPE | 50,257 |
| GPT-3 | BPE | 50,257 |
| GPT-4 | Tiktoken (BPE) | ~100,000+ |
| BERT | WordPiece | 30,522 |
| Llama 2 | SentencePiece | 32,000 |
| Llama 3 | Tiktoken | 128,256 |
| Mistral 7B | SentencePiece | 32,000 |
| Claude | Custom BPE | ~100,000+ |

**Impact of vocabulary size:**

| | Small Vocabulary (32K) | Large Vocabulary (100K+) |
|---|---|---|
| Embedding matrix size | Smaller | Larger |
| Memory usage | Less | More |
| Rare word coverage | More splitting | More intact tokens |
| Non-English languages | More token splits | More efficient |
| Sequence length | Longer (more splits) | Shorter (fewer splits) |
| Training data needed | Less | More |

**Why Llama 3 expanded from 32K to 128K vocabulary:**
Llama 2's 32K vocabulary was too small for multilingual and code tasks — non-English languages required far more tokens per sentence. The 128K vocabulary means each token represents more information, sequences are shorter, and the model is more efficient across languages.

**The embedding matrix cost:**
```
Vocabulary Size × Embedding Dimension = Embedding Matrix Parameters

GPT-2:   50,257 × 768   = 38.6M parameters
Llama 3: 128,256 × 4096 = 525M parameters  ← just for embeddings!
```

---

## 17. Special Tokens

Special tokens are reserved vocabulary entries that carry structural meaning — they are control signals, not natural language.

| Token | Name | Purpose |
|---|---|---|
| `<BOS>` or `<s>` | Beginning of Sequence | Marks the start of a sequence. Signals to the model where input begins. |
| `<EOS>` or `</s>` | End of Sequence | Tells the model to stop generating. Without this, generation would never end. |
| `<PAD>` | Padding | Pads shorter sequences to match the longest in a batch (for efficient GPU processing). |
| `<UNK>` | Unknown | Represents any token not in the vocabulary. Rare in subword models since anything can be decomposed. |
| `<SEP>` | Separator | Separates distinct segments (e.g., question from context in BERT). |
| `<MASK>` | Mask | Used during BERT-style masked language model training. The model predicts what was masked. |
| `[INST]` | Instruction | Llama's delimiter for instruction boundaries in chat templates. |
| `<\|im_start\|>` | Message Start | ChatML format used by OpenAI and others. Marks the start of a message. |
| `<\|im_end\|>` | Message End | ChatML format. Marks the end of a message. |

**Why special tokens matter for engineering:**

Each model family has its own chat template using different special tokens. If you use the wrong template, the model doesn't know where instructions end and user input begins — output quality degrades significantly and silently.

**Llama 3 chat template example:**
```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is Python?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

**GPT / ChatML template example:**
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is Python?<|im_end|>
<|im_start|>assistant
```

**Production gotcha:** When using open-source models via Ollama, vLLM, or HuggingFace, always use the model's built-in chat template method rather than formatting strings manually:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is Python?"}
]

# Correct — uses the model's built-in template
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
```

---

## 18. Token IDs

A **Token ID** is the integer that uniquely identifies a token in the vocabulary. It is what actually gets passed into the model — the model never sees text, only token IDs.

```
Vocabulary:
Token "I"       → ID 40
Token " love"   → ID 1842
Token " Python" → ID 11361

Sentence: "I love Python"
Token IDs: [40, 1842, 11361]
```

The token ID is the index into the embedding matrix. When the model receives token ID 1842, it looks up row 1842 in the embedding matrix and retrieves the corresponding 768-dimensional (or 4096-dimensional) vector.

```
Embedding Matrix (50,257 × 768):
Row 0:    [0.12, -0.34, ...]   ← token ID 0
Row 1:    [0.56,  0.78, ...]   ← token ID 1
...
Row 1842: [0.67,  0.12, ...]   ← token ID 1842 (" love")
...
```

---

## 19. Token IDs vs Tokens

| Aspect | Token | Token ID |
|---|---|---|
| What it is | A piece of text | An integer |
| Example | " love" | 1842 |
| Human-readable | Yes | No |
| What the model uses | No | Yes |
| Stored in | Vocabulary | Embedding matrix index |

**The flow:**
```
Text → [Tokenizer] → Tokens → [Vocabulary lookup] → Token IDs → [Embedding] → Vectors → [Model]
```

**Important:** The same text can produce different token IDs depending on which tokenizer (and which model) you use. Token ID 1842 in GPT-2 is not the same as token ID 1842 in Llama 3. Tokenizers are model-specific.

---

## 20. Tokenization Pipeline

The complete tokenization pipeline from raw text to model-ready input:

```
Step 1: Normalisation
   "Hello  World!" → "Hello World!"
   (lowercase, remove extra spaces, unicode normalisation)

Step 2: Pre-tokenization
   "Hello World!" → ["Hello", "World", "!"]
   (split on whitespace/punctuation before subword splitting)

Step 3: Subword Tokenization (BPE/WordPiece/SentencePiece)
   ["Hello", "World", "!"] → ["Hello", "▁World", "!"]

Step 4: Vocabulary Lookup
   ["Hello", "▁World", "!"] → [15496, 1024, 0]

Step 5: Special Token Addition
   [15496, 1024, 0] → [<BOS>, 15496, 1024, 0, <EOS>]
   (model-specific, depends on task)

Step 6: Padding (for batches)
   Sequence 1: [<BOS>, 15496, 1024, 0, <EOS>, <PAD>, <PAD>]
   Sequence 2: [<BOS>, 40, 1842, 11361, 502, <EOS>, <PAD>]
   (padded to same length for batch GPU processing)

Output: Tensor of integer token IDs ready for embedding lookup
```

---

## 21. What Happens After Tokenization?

After tokenization produces token IDs, the model pipeline continues:

```
Token IDs: [40, 1842, 11361]
    ↓
Embedding Lookup
    → Each ID retrieves its row from the embedding matrix
    → Output: matrix of shape [sequence_length × embedding_dim]
    → [3 × 768] for 3 tokens with 768-dim embeddings

    ↓
Positional Encoding Added
    → Adds position information to each token's vector
    → Model now knows "I" is position 0, "love" is position 1, etc.

    ↓
Transformer Blocks (stacked N times)
    → Self-attention, FFN, Add&Norm

    ↓
Linear Layer + Softmax
    → Produces probability distribution over all 50,257 vocabulary tokens

    ↓
Token Sampling
    → Pick next token (greedy, top-k, top-p, temperature)

    ↓
Detokenization
    → Convert token ID back to text
    → Append to output
    → Repeat from beginning until <EOS>
```

---

## 22. Common Misconceptions about Tokens

**Misconception 1: "1 token = 1 word"**
False. "Tokenization" = 3 tokens. "ChatGPT" = 3 tokens. "the" = 1 token. The relationship is approximate: 1 token ≈ 0.75 words in English.

**Misconception 2: "Tokenization is the same for all models"**
False. Every model family has its own tokenizer. Token ID 42 in GPT-2 is "I". Token ID 42 in Llama is something completely different. You cannot mix tokenizers.

**Misconception 3: "Longer words always cost more tokens"**
Not necessarily. Common long words like "learning" or "programming" may be a single token. Rare short words may split. "ChatGPT" is rarer than "learning" so it splits more.

**Misconception 4: "The model understands the tokens it processes"**
The model processes token IDs mapped to vectors. It has no inherent "understanding" — understanding emerges from learned patterns in the weights. A token is just a number.

**Misconception 5: "Non-English text uses the same number of tokens as English"**
False — and this matters for cost. Languages with characters outside the ASCII range often require more tokens. Chinese, Japanese, Arabic, and Hindi text typically requires 2–5x more tokens per concept than English. Some tokenizers (like Llama 3's 128K vocabulary) are better at this than older models.

```
"Hello" (English) → 1 token
"こんにちは" (Japanese, "hello") → 3–5 tokens depending on tokenizer
```

**Misconception 6: "Output tokens are the same price as input tokens"**
False. Output tokens are almost always more expensive (typically 2–5x). More on this in Section D.

---

# B. Token Count

---

## 23. What is Token Count?

**Token count** is simply the total number of tokens in a piece of text — input, output, or combined.

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")

text = "I love building AI systems with Python and FastAPI"
tokens = enc.encode(text)
print(f"Token count: {len(tokens)}")   # 10

# For an API call
input_tokens = len(enc.encode(system_prompt + user_message))
output_tokens = len(enc.encode(response))
total_tokens = input_tokens + output_tokens
```

Token count is the fundamental unit for measuring everything in LLM systems: cost, latency, context usage, and quality.

---

## 24. Why Does Token Count Matter?

Token count is to LLM systems what bytes are to storage or requests-per-second are to web APIs — the primary resource metric that everything else derives from.

| What you care about | Driven by token count |
|---|---|
| Cost per API call | Input tokens × input price + Output tokens × output price |
| Response latency | Output tokens generated × time per token |
| Context window usage | Sum of all tokens in the current context |
| Model capability | Whether your content fits in the context window at all |
| Scalability | Total tokens per day × cost per token |

---

## 25–30. How Token Count Affects Everything

### 25. Cost

Direct linear relationship. Every token costs money.

```
Cost = (input_tokens × input_price_per_1M) + (output_tokens × output_price_per_1M)

Example: GPT-4o
  Input:  $2.50 per 1M tokens
  Output: $10.00 per 1M tokens

  1000-token prompt + 500-token response:
  Cost = (1000 × 2.50/1,000,000) + (500 × 10.00/1,000,000)
       = $0.0025 + $0.005
       = $0.0075 per call

  At 100,000 calls/month:
  Monthly cost = $750
```

A bloated system prompt that's 2,000 tokens instead of 500 tokens costs 4x more at the same request volume — and that's just on the input side.

### 26. Latency

Output tokens are generated sequentially — one at a time. Each token takes compute time (a forward pass through the model). More output tokens = longer response time, linearly.

```
Time to first token (TTFT):  ~0.5–2 seconds (processing input)
Time per output token:        ~20–100ms per token (model dependent)

100-token response:  TTFT + 100 × 50ms = ~5.5 seconds
500-token response:  TTFT + 500 × 50ms = ~25.5 seconds
```

This is why streaming exists — you show tokens as they arrive instead of waiting for the full response.

### 27. Context Usage

Every token in the current context window — system prompt, conversation history, retrieved documents, tool results — consumes context capacity. When you run out, either the oldest content gets truncated or the API returns an error.

```
Context Window: 128,000 tokens

System prompt:           2,000 tokens
Conversation history:   15,000 tokens
Retrieved RAG documents: 8,000 tokens
User message:              500 tokens
─────────────────────────────────────
Total input used:        25,500 tokens
Remaining for output:   102,500 tokens
```

Managing what goes into context is one of the primary engineering challenges in production AI systems.

### 28. Scalability

At small scale, token inefficiency is invisible. At production scale, it's a cost multiplier that determines whether a product is profitable.

```
Baseline: 1,000 tokens per request, 100,000 requests/day
Tokens/day: 100M tokens

After optimisation: 600 tokens per request (40% reduction)
Tokens/day: 60M tokens
Cost saving: 40% on token costs
At $5/1M tokens: saves $200/day = $6,000/month
```

### 29. Response Quality

More tokens ≠ better quality. Padding responses with filler tokens reduces signal-to-noise ratio. But too few tokens forces the model to truncate reasoning.

The optimal output length depends entirely on the task:
- Classification: 1–5 tokens
- Summarisation: 100–500 tokens
- Code generation: 200–2000 tokens
- Essay writing: 500–3000 tokens

Forcing longer outputs with `min_tokens` often results in repetition and padding. Forcing shorter outputs with aggressive `max_tokens` truncates reasoning. Set `max_tokens` to a reasonable ceiling for your task, not to artificially cap costs.

### 30. Impact on Scalability in Production

```
Production concerns driven by token count:

Request-level:  Cost per call, latency per call
Session-level:  Context window management, conversation history trimming
Day-level:      Total spend, token budget alerting
Month-level:    Infrastructure cost, model selection decisions
```

---

## 31. Why AI Engineers Care About Token Count

As a backend engineer, you optimise for throughput, latency, and infrastructure cost. Token count is the equivalent metric in the AI engineering world — it's the resource you're constantly balancing.

**Specific concerns for AI engineers:**

- **Prompt design:** A 3,000-token system prompt costs 3x more than a 1,000-token one, at every single API call
- **History management:** Conversation history grows with each turn — you need a strategy to trim or summarise it
- **RAG design:** How many chunks to retrieve and how large each chunk is directly impacts both cost and quality
- **Model selection:** GPT-4o at $10/1M output tokens vs GPT-4o-mini at $0.60/1M — same task, 16x cost difference
- **Caching:** Identical prompts can be cached; Anthropic's prompt caching can save 90% on repeated system prompts

---

# C. Context Window

---

## 32. What is a Context Window?

The **context window** is the maximum number of tokens an LLM can process in a single interaction — the combined total of everything it can "see" at once: system prompt + conversation history + retrieved documents + user input + its own output.

Think of the context window as the model's working memory. Everything inside it is "known" to the model during this call. Everything outside it simply does not exist from the model's perspective.

```
Context Window = 128,000 tokens (GPT-4o)
│
├── System Prompt          2,000 tokens
├── Conversation History  15,000 tokens
├── Retrieved Documents    8,000 tokens
├── User Message             500 tokens
└── [Reserved for Output] 102,500 tokens available
```

---

## 33. Why do LLMs need a Context Window?

The context window is not a design choice — it is a mathematical constraint of the Transformer architecture.

Self-attention, the core mechanism in Transformers, computes relationships between every pair of tokens. To do this, the model must hold representations of all tokens simultaneously in memory.

There is no "scrolling" or "paging" — it's all-or-nothing. Either a token is in the context window and the model can attend to it, or it isn't and the model has no knowledge of it.

Without a context window limit, self-attention would require unlimited memory and unlimited computation — neither of which is physically possible.

---

## 34. What Contributes to the Context Window?

Every token in the following list is counted against your context window budget:

| Component | Typical Token Count | Notes |
|---|---|---|
| System prompt | 200–5,000 | Fixed cost on every call |
| Few-shot examples | 500–3,000 | Each example costs tokens |
| Conversation history | Grows with turns | Major source of context bloat |
| Retrieved documents (RAG) | 1,000–20,000 | Largest variable cost |
| Tool/function definitions | 200–2,000 | Each tool definition adds tokens |
| Tool call results | 100–5,000 | Returned data costs tokens |
| Current user message | 10–1,000 | Usually small |
| Model's own output | 100–4,000 | Counted in output tokens |

**The common trap:** Developers focus on the user message and forget that the system prompt, history, and RAG content are all consuming context — sometimes leaving very little room for the model to respond.

---

## 35. Input Tokens vs Output Tokens within Context

The context window budget is split between input and output — but not equally. You control how much you allocate.

```
Total context window: 128,000 tokens
│
├── Input tokens (everything you send):
│     System prompt + history + documents + user message
│     You control this
│
└── Output tokens (what the model generates):
      The model's response
      Controlled by max_tokens parameter
      Must fit within remaining context budget
```

**Key rule:** `input_tokens + output_tokens ≤ context_window_size`

If your input is 120,000 tokens and the context window is 128,000, the model can generate at most 8,000 tokens in response — regardless of what you set `max_tokens` to.

---

## 36. Context Window Sizes of Popular Models

| Model | Context Window | Notes |
|---|---|---|
| GPT-3.5-turbo | 16,384 tokens | Legacy, mostly replaced |
| GPT-4o | 128,000 tokens | Standard large model |
| GPT-4o-mini | 128,000 tokens | Cheaper, smaller |
| o1, o3 | 200,000 tokens | Reasoning models |
| Claude 3 Haiku | 200,000 tokens | Fast, cheap |
| Claude 3.5 Sonnet | 200,000 tokens | Best balance |
| Claude 3 Opus | 200,000 tokens | Most capable |
| Gemini 1.5 Pro | 1,000,000 tokens | 1M context |
| Gemini 1.5 Flash | 1,000,000 tokens | Faster, cheaper |
| Llama 3.1 8B | 128,000 tokens | Open source |
| Llama 3.1 70B | 128,000 tokens | Open source |
| Llama 3.1 405B | 128,000 tokens | Open source |
| Mistral 7B | 32,768 tokens | Smaller model |

**Rough content sizes for reference:**
```
4,000 tokens   ≈ a short article (~3,000 words)
16,000 tokens  ≈ a long technical blog post or chapter
32,000 tokens  ≈ a short story or academic paper
128,000 tokens ≈ a short novel (~96,000 words)
200,000 tokens ≈ a full-length novel
1,000,000 tokens ≈ ~750,000 words (several books)
```

---

## 37. Why is the Context Window Limited?

Three reasons: memory, computation, and training.

**Memory:**
During a forward pass, the model must store KV (Key-Value) matrices for every token in the context. For a 70B parameter model with 128K context:
```
KV cache size = context_length × layers × heads × head_dim × 2 × precision
= 128,000 × 80 × 64 × 128 × 2 × 2 bytes ≈ hundreds of GB
```
More context = more GPU memory required per inference request.

**Computation:**
Self-attention is O(n²) in sequence length. Double the context → 4x the attention computation.
```
4K context:   4,000²  = 16M attention operations
128K context: 128,000² = 16.4B attention operations (1,024x more!)
```

**Training:**
Models must be trained on examples up to their maximum context length. Training on long sequences is expensive — you need examples, compute, and time proportional to that length.

Modern techniques like FlashAttention, sparse attention, and ring attention help push context limits upward, but the fundamental O(n²) constraint still dominates at extreme lengths.

---

## 38. Self-Attention Complexity and Context Limits

Self-attention computes a score between every pair of tokens. For n tokens:

```
Number of attention pairs = n × n = n²

n = 1,000:   1,000,000 pairs
n = 4,000:   16,000,000 pairs
n = 16,000:  256,000,000 pairs
n = 128,000: 16,384,000,000 pairs  ← 16 billion pairs per attention head
```

Each pair requires storing a Key and Value vector (the KV cache). At 128K context across 80 transformer layers, this becomes the dominant memory consumer during inference.

**This is why larger context windows require:**
- More GPU memory per request
- Higher latency
- Higher cost per token
- More expensive hardware

**Engineering implication:** Don't use the full context window just because it's available. Filling 128K context costs ~32x more in attention computation than filling 4K context. Be selective about what you include.

---

## 39. Advantages of Large Context Windows

**1. Long document processing without chunking**
A 200K context window lets you pass an entire 150-page technical document in one call — no chunking, no retrieval errors, no lost context.

**2. Longer conversation history**
No need to truncate or summarise old turns. The model remembers the full conversation.

**3. Complex multi-document tasks**
Compare, synthesise, or cross-reference multiple long documents in a single call.

**4. Reduced RAG complexity**
For small corpora (tens of documents), you can skip RAG entirely and just send everything directly.

**5. Better coherence in long outputs**
When generating long documents (reports, books, code), the model can reference earlier sections it generated.

---

## 40. Disadvantages of Large Context Windows

**1. Cost**
More input tokens = higher API cost. Sending 100K tokens of context on every request can be very expensive.

**2. Latency**
Processing a 100K token input takes significantly longer than a 2K input.

**3. Lost in the Middle problem**
LLMs pay less attention to information in the middle of a very long context. Relevant information buried 50K tokens in may be effectively ignored. (Covered in detail in Section F.)

**4. Quality degradation**
More context ≠ better answers. Irrelevant information in context can confuse the model and reduce answer quality.

**5. Memory requirements**
Self-hosted models with large contexts require proportionally more GPU memory per concurrent request.

---

## 41. Advantages of Small Context Windows

- Lower cost per call
- Lower latency
- Forces discipline in prompt design — you think carefully about what to include
- Easier to debug (less content to reason about)
- Lower memory requirements for self-hosted models

---

## 42. Disadvantages of Small Context Windows

- Cannot process long documents without chunking
- Conversation history must be actively managed and trimmed
- RAG is required for any knowledge base larger than the context window
- Multi-step reasoning tasks may not fit

---

## 43. Context Window vs Memory

This is one of the most common misconceptions among developers new to LLMs.

| | Context Window | Memory |
|---|---|---|
| What it is | Active working space for the current call | Persistent storage across calls |
| Scope | Single API call only | Across sessions, days, weeks |
| Storage | GPU memory (KV cache) | External database |
| Survives after call | No — wiped completely | Yes |
| Capacity | Thousands–millions of tokens | Unlimited (database) |
| Access pattern | Everything available simultaneously | Retrieved via search |

**Critical point:** LLMs have NO built-in memory. When an API call ends, everything in the context window is gone. The next call starts completely fresh.

If a ChatGPT conversation "remembers" what you said yesterday, it's because the application layer retrieved that information from a database and injected it into the new context — not because the model remembered.

---

## 44. Context Window vs Long-Term Memory

Long-term memory in AI applications is always implemented externally — the LLM itself has no native long-term memory.

**Pattern for long-term memory:**

```
User Message
     ↓
Retrieve relevant past memories (from vector DB or database)
     ↓
Inject relevant memories into context:
  "Previously, user mentioned they prefer Python over JavaScript"
  "Last session, user was building a FastAPI application"
     ↓
LLM processes with context of relevant memories
     ↓
Extract new information from this conversation
     ↓
Store in memory database for future retrieval
```

**Types of external memory stores:**
- **Key-value store (Redis):** Fast retrieval of specific facts by key
- **Vector database:** Semantic search over past conversations
- **Relational DB:** Structured user preferences and history
- **Episodic store:** Summaries of past conversations

The context window is the RAM. External memory is the hard drive. The application layer is the OS that decides what to load from disk into RAM.

---

## 45. Context Overflow / Exceeding Context Limits

When the total tokens (input + output) exceed the model's context window, one of the following happens:

**API returns an error:**
```json
{
  "error": {
    "message": "This model's maximum context length is 128000 tokens. 
                However, your messages resulted in 130542 tokens.",
    "type": "invalid_request_error",
    "code": "context_length_exceeded"
  }
}
```

**Automatic truncation (some implementations):**
The oldest messages are silently dropped until the content fits. This can cause the model to lose important context without any warning.

**Prevention strategies:** (covered in the next sections)

**Backend engineer parallel:** Context overflow is like a stack overflow — you've exceeded the fixed memory budget for the current execution context. The solution is the same: manage what you push onto the stack, and clean up when you're done.

---

## 46. Context Management Strategies

Managing what goes into the context window is one of the most important engineering skills in production AI systems.

**Strategy 1 — Token budgeting:**
Allocate a token budget to each component and enforce it strictly.

```python
CONTEXT_BUDGET = 128_000
SYSTEM_PROMPT_BUDGET = 2_000
HISTORY_BUDGET = 20_000
RAG_BUDGET = 40_000
OUTPUT_RESERVE = 4_000

available_for_user = CONTEXT_BUDGET - SYSTEM_PROMPT_BUDGET - HISTORY_BUDGET - RAG_BUDGET - OUTPUT_RESERVE
# = 62,000 tokens for user message
```

**Strategy 2 — Recency-weighted history:**
Keep recent conversation turns in full; summarise or drop older turns.

```python
def manage_history(history, max_tokens=20_000):
    # Always keep the last 4 turns
    recent = history[-4:]
    older = history[:-4]
    
    recent_tokens = count_tokens(recent)
    remaining = max_tokens - recent_tokens
    
    if count_tokens(older) > remaining:
        # Summarise older history
        summary = summarise(older)
        return [{"role": "system", "content": f"Earlier context: {summary}"}] + recent
    
    return older + recent
```

**Strategy 3 — Selective RAG injection:**
Only inject the most relevant retrieved chunks, ranked by relevance score.

```python
def select_chunks(retrieved_chunks, token_budget=40_000):
    selected = []
    used = 0
    for chunk in retrieved_chunks:  # already sorted by relevance
        chunk_tokens = count_tokens(chunk.text)
        if used + chunk_tokens > token_budget:
            break
        selected.append(chunk)
        used += chunk_tokens
    return selected
```

**Strategy 4 — Dynamic truncation:**
Check total token count before each API call and truncate the least important components first.

```python
def build_context(system, history, docs, query):
    while count_total_tokens(system, history, docs, query) > MAX_TOKENS:
        if len(docs) > 0:
            docs = docs[:-1]           # drop least relevant doc first
        elif len(history) > 2:
            history = history[2:]      # drop oldest history turn
        else:
            break                      # nothing more to drop
    return assemble(system, history, docs, query)
```

---

## 47. Context Compression

**Context compression** reduces the token count of existing content while preserving its essential information — making the same content fit in a smaller token budget.

**Technique 1 — LLMLingua (selective token removal):**
A small trained model scores each token for importance and removes low-importance tokens. Can achieve 5–20x compression with minimal quality loss.

```
Original: "The patient was admitted to the hospital on Monday with
           severe chest pain. Tests were conducted. The results
           showed elevated troponin levels indicating a possible
           myocardial infarction event."

Compressed: "Patient admitted Monday, severe chest pain. Tests:
             elevated troponin, possible myocardial infarction."
```

**Technique 2 — AutoCompressor:**
Fine-tuned model that compresses long contexts into a compact "summary token" vector that the main model can condition on.

**Technique 3 — Structured compression:**
Domain-specific compression rules. For code: remove comments and whitespace. For JSON: remove null values and redundant keys. For conversations: remove filler phrases ("Um, well, I think that...").

**When to use:** When you have more relevant content than fits in the context window and summarisation would lose too much detail.

---

## 48. Context Summarisation

**Context summarisation** condenses conversation history or retrieved documents into a shorter summary, freeing up context window space.

**When history grows too long:**

```python
async def summarise_old_history(old_turns: list) -> str:
    summary_prompt = f"""
    Summarise the following conversation history concisely.
    Preserve: key facts mentioned, decisions made, user preferences stated.
    Remove: pleasantries, repetitions, tangents.
    
    Conversation:
    {format_turns(old_turns)}
    
    Summary (max 300 words):
    """
    response = await llm.complete(summary_prompt)
    return response.text
```

**Trade-off:** Summarisation loses detail. Information that seems unimportant now may matter later. Use it for old history (many turns back) but keep recent turns verbatim.

**Hierarchical summarisation for very long conversations:**
```
Turn 1–20:   Summarised into "Summary Block 1"
Turn 21–40:  Summarised into "Summary Block 2"
Turn 41–50:  Kept verbatim (recent)

Context:
[Summary Block 1] + [Summary Block 2] + [Turns 41-50] + [Current message]
```

---

## 49. Sliding Window Approach

The **sliding window** approach processes long documents by moving a fixed-size window over the content, always keeping the most recent or most relevant portion in context.

**Use case:** Chatting with a very long document (book, large codebase, long report) that doesn't fit in the context window.

**Basic sliding window:**
```
Document: [Chunk1][Chunk2][Chunk3][Chunk4][Chunk5]

Window size: 3 chunks

Step 1: Process [Chunk1][Chunk2][Chunk3]
Step 2: Process [Chunk2][Chunk3][Chunk4]  ← overlap maintains continuity
Step 3: Process [Chunk3][Chunk4][Chunk5]
```

**Overlap** is crucial — without it, information at chunk boundaries is lost.

**Limitations:** The model cannot connect information from Chunk1 to Chunk5 in a single pass. For tasks that require global understanding across a document (summarisation, Q&A over entire books), RAG or hierarchical summarisation is better.

---

# D. Input & Output Tokens

---

## 50. What are Input Tokens?

**Input tokens** are all the tokens you send to the model in a single API call — everything that forms the prompt.

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a Python expert."},   # ← input tokens
        {"role": "user", "content": "Explain list comprehensions."}   # ← input tokens
    ]
)
# response.usage.prompt_tokens = total input tokens
```

---

## 51. What are Output Tokens?

**Output tokens** are the tokens generated by the model in its response — everything the model writes back.

```python
response = client.chat.completions.create(...)
print(response.choices[0].message.content)  # ← these are the output tokens
print(response.usage.completion_tokens)      # count of output tokens
```

---

## 52. Input Tokens vs Output Tokens

| Aspect | Input Tokens | Output Tokens |
|---|---|---|
| Direction | You → Model | Model → You |
| Generated by | You (the application) | The model |
| Processing | Processed in parallel (one forward pass) | Generated sequentially (one token at a time) |
| Pricing | Lower (typically) | Higher (typically 2–5x) |
| Affects latency | Slightly (longer input = slower first token) | Strongly (more output = proportionally longer) |
| Cacheable | Yes (prompt caching) | No |

---

## 53. What Counts as Input Tokens?

Everything you send in the API request:

```
Input tokens include:
├── System message text
├── All previous assistant messages (conversation history)
├── All previous user messages (conversation history)
├── Current user message
├── Retrieved context/documents injected into the prompt
├── Tool/function definitions
├── Results from previous tool calls
├── Few-shot examples
├── Special tokens and delimiters
└── Images (counted as equivalent tokens in multimodal models)
```

**Image token counting:**
Images in multimodal models (GPT-4V, Claude Vision) are also counted as tokens. A 1024×1024 image can cost 765–1,105 tokens in GPT-4V depending on detail level. This is significant when processing many images.

---

## 54. What Counts as Output Tokens?

Everything the model generates in its response:

```
Output tokens include:
├── The model's text response
├── Generated code
├── Reasoning/thinking tokens (in reasoning models — sometimes visible)
├── Tool call JSON (when model decides to call a function)
└── Special tokens marking end of response
```

**Reasoning model note:** Models like o1 and o3 generate internal "thinking" tokens that are invisible to you but still counted and billed as output tokens. A single o1 response may generate 5,000+ internal reasoning tokens before producing the visible answer.

---

## 55. Why are Output Tokens Usually More Expensive?

Output tokens cost more because generating them is computationally more intensive than processing input tokens.

**Input processing (one forward pass):**
All input tokens are processed in parallel in a single forward pass through the model. The computation scales linearly with sequence length.

**Output generation (sequential forward passes):**
Each output token requires a complete separate forward pass through the entire model. To generate 500 output tokens, the model runs 500 forward passes.

```
Input processing:
[T1, T2, T3, ..., T1000] → one forward pass → processed simultaneously

Output generation:
[T1001] → forward pass 1 → token generated
[T1001, T1002] → forward pass 2 → token generated
[T1001, T1002, T1003] → forward pass 3 → token generated
...500 forward passes for 500 output tokens
```

**Pricing examples:**

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Ratio |
|---|---|---|---|
| GPT-4o | $2.50 | $10.00 | 4x |
| GPT-4o-mini | $0.15 | $0.60 | 4x |
| Claude 3.5 Sonnet | $3.00 | $15.00 | 5x |
| Claude 3 Haiku | $0.25 | $1.25 | 5x |
| Gemini 1.5 Flash | $0.075 | $0.30 | 4x |

---

## 56. How Output Tokens are Generated Sequentially

Understanding sequential generation is key to understanding latency in LLM systems.

```
Step 1: Model processes entire input (one pass)
        → Generates probability distribution over vocabulary
        → Samples token: "The"

Step 2: Input + "The" → Model forward pass
        → Samples token: "capital"

Step 3: Input + "The capital" → Model forward pass
        → Samples token: "of"

Step 4: Input + "The capital of" → Model forward pass
        → Samples token: "France"

Step 5: Input + "The capital of France" → Model forward pass
        → Samples token: "is"

...continues until <EOS> or max_tokens reached
```

**KV Cache:** Without optimisation, this would be extremely slow — each step re-processes all previous tokens from scratch. The KV Cache stores the Key and Value matrices from previous forward passes so they don't need recomputation. Each new step only computes the KV matrices for the newly added token.

**Latency model:**
```
Total latency = TTFT + (output_tokens × time_per_token)

Where:
  TTFT = Time to First Token (input processing time)
  time_per_token = time for one forward pass ≈ 20–100ms depending on model and hardware
```

---

## 57. Input vs Output Trade-offs

**When to prefer more input tokens:**
- Complex task instructions that would otherwise result in errors or poor output
- RAG context that grounds the answer in facts (reduces hallucinations)
- Few-shot examples that enforce format consistency
- Conversation history needed for coherence

**When to prefer more output tokens:**
- Complex reasoning tasks (Chain of Thought needs space)
- Code generation (complete, working code)
- Detailed analysis or reports

**When to minimise both:**
- High-volume production (cost-sensitive)
- Real-time applications (latency-sensitive)
- Classification tasks (output is always 1–5 tokens)

**Engineering principle:** Every token in your prompt must earn its place. If removing a sentence doesn't change output quality, remove it.

---

# E. API Costing

---

## 58. How API Pricing Works

LLM providers charge per token consumed — both for what you send (input) and what you receive (output). Prices are quoted per 1 million tokens (per 1M).

This is a pay-per-use model with no fixed cost, which is excellent for development but requires careful management at production scale.

---

## 59. Pricing Based on Token Usage

Providers give you token usage in every API response:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    max_tokens=500
)

usage = response.usage
print(f"Input tokens:  {usage.prompt_tokens}")      # e.g., 1,243
print(f"Output tokens: {usage.completion_tokens}")  # e.g., 387
print(f"Total tokens:  {usage.total_tokens}")       # e.g., 1,630

# Anthropic SDK
response = client.messages.create(...)
print(response.usage.input_tokens)
print(response.usage.output_tokens)
```

Always log these values. Without them, you cannot estimate costs or detect anomalies.

---

## 60. Input Pricing vs Output Pricing

| Provider | Model | Input (per 1M) | Output (per 1M) |
|---|---|---|---|
| OpenAI | GPT-4o | $2.50 | $10.00 |
| OpenAI | GPT-4o-mini | $0.15 | $0.60 |
| OpenAI | o3-mini | $1.10 | $4.40 |
| Anthropic | Claude 3.5 Sonnet | $3.00 | $15.00 |
| Anthropic | Claude 3 Haiku | $0.25 | $1.25 |
| Google | Gemini 1.5 Pro | $1.25 | $5.00 |
| Google | Gemini 1.5 Flash | $0.075 | $0.30 |
| Groq | Llama 3.1 70B | $0.59 | $0.79 |

*Prices as of mid-2025. Always verify with provider documentation.*

---

## 61. API Cost Formula

```
Cost per call = (input_tokens / 1,000,000 × input_price) 
              + (output_tokens / 1,000,000 × output_price)

Daily cost    = calls_per_day × average_cost_per_call

Monthly cost  = daily_cost × 30
```

**Worked example — RAG-based internal Q&A tool:**

```
Per call breakdown:
  System prompt:        800 tokens
  Conversation history: 2,000 tokens
  RAG documents:        5,000 tokens
  User message:         200 tokens
  ─────────────────────────────────
  Total input:          8,000 tokens

  Average response:     600 tokens output

GPT-4o pricing:
  Input cost:  8,000 / 1,000,000 × $2.50  = $0.020
  Output cost:   600 / 1,000,000 × $10.00 = $0.006
  Cost per call: $0.026

At 500 calls/day:
  Daily cost:   $13.00
  Monthly cost: $390.00

After optimisation (cut RAG to 2,000 tokens, trim history to 1,000):
  New input: 4,000 tokens
  New cost per call: $0.010 + $0.006 = $0.016 (38% reduction)
  Monthly cost: $240.00  ← saves $150/month
```

---

## 62. Factors Affecting API Cost

| Factor | Impact | Control |
|---|---|---|
| Model choice | 10–100x cost difference | Choose smallest model that meets quality bar |
| Input token count | Linear | Optimise prompts, trim history, reduce RAG chunks |
| Output token count | Linear (but 4–5x per token) | Set appropriate max_tokens, use concise output instructions |
| Caching | Up to 90% reduction on repeated input | Use prompt caching for static system prompts |
| Batch vs real-time | 50% reduction for batch | Use batch API for offline workloads |
| Request volume | Linear | Cache responses for identical queries |
| Image inputs | High (images = many tokens) | Resize/downsample images before sending |

---

## 63. Cost Estimation in Production

Before going to production, estimate your monthly cost:

```python
def estimate_monthly_cost(
    calls_per_day: int,
    avg_input_tokens: int,
    avg_output_tokens: int,
    input_price_per_1m: float,
    output_price_per_1m: float
) -> dict:
    
    cost_per_call = (
        (avg_input_tokens / 1_000_000 * input_price_per_1m) +
        (avg_output_tokens / 1_000_000 * output_price_per_1m)
    )
    
    daily_cost = calls_per_day * cost_per_call
    monthly_cost = daily_cost * 30
    
    return {
        "cost_per_call": round(cost_per_call, 6),
        "daily_cost": round(daily_cost, 2),
        "monthly_cost": round(monthly_cost, 2),
        "tokens_per_day": calls_per_day * (avg_input_tokens + avg_output_tokens)
    }

# GPT-4o example
estimate = estimate_monthly_cost(
    calls_per_day=1000,
    avg_input_tokens=5000,
    avg_output_tokens=500,
    input_price_per_1m=2.50,
    output_price_per_1m=10.00
)
print(estimate)
# {'cost_per_call': 0.0175, 'daily_cost': 17.50, 'monthly_cost': 525.0, ...}
```

---

## 64. Token Usage Monitoring

In production, monitor token usage per endpoint, per user, and per model to catch anomalies and optimise spend.

**Minimum logging you should implement:**

```python
import logging
from functools import wraps

def track_token_usage(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        response = await func(*args, **kwargs)
        
        usage = response.usage
        logging.info({
            "event": "llm_call",
            "model": response.model,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "estimated_cost_usd": calculate_cost(usage),
            "endpoint": kwargs.get("endpoint", "unknown"),
            "user_id": kwargs.get("user_id", "anonymous")
        })
        
        return response
    return wrapper
```

**What to alert on:**
- Single call input_tokens > 50K (likely a prompt injection or infinite loop)
- Daily cost exceeds threshold
- Sudden spike in output tokens per call (model becoming verbose)
- p95 latency exceeds SLA (usually correlated with output token count)

---

## 65. Cost Optimization Techniques

**1. Use the smallest model that meets quality requirements**

For most production tasks, GPT-4o-mini or Claude Haiku is sufficient. Use large models only for complex reasoning.

```python
def select_model(task_complexity: str) -> str:
    if task_complexity == "simple":    # classification, extraction
        return "gpt-4o-mini"          # $0.15/$0.60 per 1M
    elif task_complexity == "medium":  # Q&A, summarisation
        return "gpt-4o-mini"          # same — mini is surprisingly capable
    elif task_complexity == "complex": # multi-step reasoning, complex code
        return "gpt-4o"               # $2.50/$10.00 per 1M
```

**2. Compress system prompts**

System prompt is paid on every single API call. A 3,000-token system prompt vs 800-token is 2,200 extra tokens × every call.

```
Before: "You are an AI assistant that helps users with Python programming questions. 
         You should always be helpful, accurate, and provide code examples when 
         relevant. Please format your code with proper syntax highlighting..."
(47 words, ~65 tokens)

After: "Python expert. Code examples when helpful. Be concise."
(8 words, ~11 tokens — same quality in most cases)
```

**3. Implement response caching**

Identical or near-identical queries should not make redundant API calls.

```python
import hashlib, json
from redis import Redis

redis = Redis()

async def cached_llm_call(messages: list, model: str) -> str:
    cache_key = hashlib.md5(
        json.dumps({"messages": messages, "model": model}).encode()
    ).hexdigest()
    
    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)
    
    response = await llm_call(messages, model)
    redis.setex(cache_key, 3600, json.dumps(response))  # 1hr TTL
    return response
```

**4. Set appropriate max_tokens**

Don't set `max_tokens=4096` for a task that needs 100 tokens. You won't get billed for ungenerated tokens, but you do get billed for context window used. More importantly, high `max_tokens` with verbose models can accidentally generate 10x more output than needed.

```python
MAX_TOKENS_BY_TASK = {
    "classification":   10,
    "entity_extraction": 200,
    "summarisation":    500,
    "q_and_a":         800,
    "code_generation": 2000,
    "long_form":       4000,
}
```

**5. Trim conversation history aggressively**

Old conversation history is one of the biggest sources of token waste.

**6. Reduce RAG chunk count and size**

Retrieve 3–5 highly relevant chunks instead of 10–20 loosely relevant ones. Quality beats quantity.

---

## 66. Cached Tokens

**Prompt caching** is a feature offered by Anthropic (and increasingly others) that caches the KV computation for repeated portions of the prompt. If your system prompt is the same across calls, you only pay full price for it once.

**Anthropic prompt caching:**

```python
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": your_large_system_context,
                    "cache_control": {"type": "ephemeral"}  # ← enable caching
                },
                {
                    "type": "text",
                    "text": user_message
                }
            ]
        }
    ]
)
```

**Cost saving:**
- Cache write: 25% more expensive than normal input (one-time)
- Cache read: 90% cheaper than normal input (every subsequent call)

**When to use:** Any content that is identical across many requests — system prompts, large document contexts, few-shot examples, tool definitions.

**Example saving:**
```
Without caching:
  10,000-token system prompt × 10,000 calls × $3/1M = $300/day

With caching (after first call):
  Write:  10,000 tokens × 1 call × $3.75/1M = $0.0375 (one-time)
  Reads:  10,000 tokens × 9,999 calls × $0.30/1M = $29.99/day

Saving: $270/day = $8,100/month
```

---

## 67. Batch Processing

**Batch APIs** allow you to send many requests at once and receive results asynchronously — at lower cost.

**OpenAI Batch API:**
- 50% cheaper than real-time API
- Results within 24 hours
- Ideal for offline workloads

```python
# Create a batch file
batch_requests = [
    {"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions",
     "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "Summarise: ..."}]}},
    {"custom_id": "req-2", "method": "POST", "url": "/v1/chat/completions",
     "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "Classify: ..."}]}},
]

# Upload and submit batch
batch = client.batches.create(
    input_file_id=upload_file(batch_requests),
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
```

**When to use batch processing:**
- Nightly document processing
- Bulk embeddings generation
- Dataset analysis and labelling
- Offline summarisation pipelines
- Any task where 24h delay is acceptable

**ETL pipeline analogy:** Batch processing in LLMs is exactly like batch ETL jobs. Instead of processing records one at a time with real-time APIs, you queue them all, process overnight, and collect results in the morning.

---

## 68. Model Selection and Cost Trade-offs

Choosing the right model for each task is the single highest-leverage cost optimisation.

```
GPT-4o:      $2.50/$10.00 per 1M   ← complex reasoning, best quality
GPT-4o-mini: $0.15/$0.60 per 1M   ← 17x cheaper, 80% as capable for most tasks
```

**Decision framework:**

| Task | Recommended Model | Reasoning |
|---|---|---|
| Text classification | GPT-4o-mini / Claude Haiku | Simple, output is 1–5 tokens |
| Entity extraction | GPT-4o-mini | Structured output, few tokens |
| FAQ answering (with RAG) | GPT-4o-mini | Most are answerable with small context |
| Complex code generation | GPT-4o | Requires deep reasoning |
| Multi-step analysis | GPT-4o / Claude Sonnet | Reasoning quality matters |
| Document summarisation | GPT-4o-mini | Mostly mechanical, not creative |
| Embeddings | text-embedding-3-small | Separate model, much cheaper |

**Routing pattern:**
```python
def route_to_model(task_type: str, complexity_score: float) -> str:
    if task_type in ["classification", "extraction", "simple_qa"]:
        return "gpt-4o-mini"
    elif complexity_score > 0.8:
        return "gpt-4o"
    else:
        return "gpt-4o-mini"
```

---

# F. Long Context Challenges

---

## 69. The "Lost in the Middle" Problem

**"Lost in the Middle"** refers to the empirically observed tendency of LLMs to pay less attention to content that appears in the middle of a long context window, compared to content at the very beginning or very end.

**The research finding (Liu et al., 2023):**
When relevant information was placed at different positions within a long context, model accuracy was:
- Beginning of context → High accuracy
- End of context → High accuracy
- Middle of context → Significantly lower accuracy

This is sometimes called the "U-shaped attention curve" — the model's effective attention forms a U shape across the context, attending strongly to the edges but weakly to the middle.

---

## 70. Why "Lost in the Middle" Happens

The exact mechanism is still researched, but the leading hypothesis involves how positional encoding and attention interact over long sequences:

**Training data distribution:** Most pre-training examples are shorter documents where the important information naturally appears at the beginning or end. The model learns these patterns.

**Attention diffusion:** In very long contexts, attention weights are spread across thousands of tokens. The signal-to-noise ratio for any individual middle token decreases.

**Recency bias:** Language models naturally predict the next token based heavily on recent context (nearby tokens). This creates a built-in recency bias that favours the end of context.

**Primacy effect:** The beginning of context gets extra attention because it's processed before any other content and often contains the system prompt or task description which anchors all subsequent processing.

---

## 71. Impact on RAG

"Lost in the Middle" directly impacts RAG quality because you're typically injecting multiple retrieved chunks into the context.

**Scenario:**
```
Context structure:
[System prompt] [Chunk 1 - very relevant] [Chunk 2 - less relevant] 
[Chunk 3 - very relevant] [Chunk 4] [Chunk 5] [User question]

"Lost in the Middle" effect:
Chunk 1 → attended well (near beginning)
Chunk 2 → attended well
Chunk 3 → attended less (middle position)  ← most relevant but ignored!
Chunk 4 → attended less
Chunk 5 → attended well (near end)

Result: The model answers based on Chunk 1, 2, 5 — misses the most
        relevant information in Chunk 3.
```

**Real-world symptom:** Users report that "the AI clearly has the right document but still gives the wrong answer." Often, the relevant information was in the middle of a long context.

---

## 72. How to Mitigate "Lost in the Middle"

**Strategy 1 — Place most important content at edges:**
Put the most critical information at the very beginning (right after the system prompt) or very end (just before the user's question).

```python
def assemble_rag_context(chunks: list, query: str) -> str:
    # Sort chunks: put highest-relevance at start and end, lower in middle
    if len(chunks) <= 2:
        return "\n".join(c.text for c in chunks)
    
    sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)
    
    # Place best chunk first, second-best last, rest in middle
    ordered = (
        [sorted_chunks[0]] +
        sorted_chunks[2:] +    # lower relevance in middle
        [sorted_chunks[1]]     # second best at end
    )
    return "\n".join(c.text for c in ordered)
```

**Strategy 2 — Reduce context size:**
Fewer, more relevant chunks means less middle — the important content stays near the edges. Retrieve 3 highly relevant chunks instead of 10 loosely relevant ones.

**Strategy 3 — Re-reading / multiple passes:**
Ask the model to re-read the context, or make two calls: one to identify relevant sections, another to answer based on those sections.

**Strategy 4 — Re-ranking:**
Use a cross-encoder re-ranker to select only the truly relevant chunks before injecting. Fewer chunks = less middle = less loss.

**Strategy 5 — Structured formatting:**
Use XML tags or headers to explicitly mark important content:
```
<critical_information>
[Most relevant chunk here]
</critical_information>

[Supporting information]

<answer_based_on_critical_information>
[User's question]
</answer_based_on_critical_information>
```

---

## 73. Why More Context Doesn't Always Improve Answers

Counterintuitively, adding more context often makes answers worse. Here's why:

**1. Signal dilution:**
If you inject 50 chunks and only 3 are relevant, the model has to find the relevant signal in 94% noise. More content = more potential for confusion.

**2. Lost in the Middle:**
Relevant content buried in the middle of a 100K-token context may as well not be there.

**3. Contradictory information:**
Multiple retrieved chunks may contain contradictory facts. More chunks = higher probability of contradiction = model gives vague or hedged answers.

**4. Model overthinking:**
With too much context, the model may spend attention on irrelevant details and produce verbose, unfocused responses.

**The optimal point:**
```
Quality
  │           ●
  │         ●   ●
  │       ●       ●
  │     ●           ●
  │   ●               ●
  └────────────────────────
    Less context    More context
    
Peak quality is usually at 3–7 highly relevant chunks,
not at maximum context stuffing.
```

---

## 74. Quality vs Quantity of Context

**Quality:** How relevant is each piece of context to the current question?
**Quantity:** How many tokens of context are you providing?

**The principle: quality beats quantity every time.**

3 chunks that are directly relevant to the question → excellent answer
15 chunks where 3 are relevant and 12 are loosely related → mediocre answer (often worse than the 3-chunk version)

**How to maximise quality:**
- Use good embedding models for retrieval
- Apply cross-encoder re-ranking to filter to truly relevant chunks
- Tune chunk size so each chunk is semantically coherent
- Add metadata filters (date, document type, source) before semantic search
- Test retrieval quality separately from answer quality (RAGAS evaluation)

---

## 75. Chunk Ordering and Reranking

**Chunk ordering** matters significantly due to the Lost in the Middle effect. How you order retrieved chunks in the context determines which ones the model pays attention to.

**Reranking** is the process of taking an initial set of retrieved candidates and re-scoring them using a more accurate (but slower) model to select the best few.

**Two-stage retrieval + reranking pipeline:**

```
Stage 1 — Coarse retrieval (fast, approximate):
  Vector similarity search → top 20 candidates
  (Fast: ~10ms, approximate: BM25 or ANN)

Stage 2 — Reranking (slower, accurate):
  Cross-encoder model evaluates each (query, chunk) pair together
  → Scores all 20 candidates
  → Select top 5 (truly relevant, not just similar)
  (Slower: ~100–300ms, more accurate)

Stage 3 — Ordering in context:
  Place top-scored chunk first
  Place second-scored chunk last
  Place others in middle (less critical)
```

**Cross-encoder vs Bi-encoder:**

| | Bi-encoder (standard embedding) | Cross-encoder (reranker) |
|---|---|---|
| How it works | Embed query and chunk separately, compare | Process query + chunk together |
| Speed | Very fast (ANN lookup) | Slower (full model forward pass) |
| Accuracy | Approximate | Much higher |
| Used for | Initial retrieval (recall) | Reranking (precision) |
| Examples | text-embedding-3-small | Cohere Rerank, BGE-Reranker |

**Popular reranking models:**
- Cohere Rerank API
- BGE-Reranker (open source, HuggingFace)
- Jina Reranker
- Voyage AI Rerank

---

# G. Production & Interview Perspectives

---

## 76. Backend Engineer Mental Models

**Tokens as bytes:**
Just as you think in bytes/KB/MB when working with data, think in tokens when working with LLMs. Your mental unit shifts from "characters" to "tokens". 1 token ≈ 4 bytes of English text.

**Context window as RAM:**
The context window is working memory — fast, limited, volatile. External databases are storage — slower, unlimited, persistent. You manage what fits in RAM exactly as you manage what fits in context.

**API cost as database query cost:**
Every LLM call is like a database query — it costs compute proportional to how much data you touch. Optimise by reducing what you send (like reducing SELECT columns), caching repeated calls (like query caching), and routing simple queries to cheaper resources (like using read replicas).

**Token counting as payload size:**
Before sending an API request, count the tokens like you'd check the payload size before an HTTP request. Set limits, monitor, and alert.

**Rate limits as throttling:**
LLM APIs have token-per-minute and request-per-minute limits. Implement exponential backoff and retry logic exactly as you would for any external API.

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai

@retry(
    retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
async def call_llm_with_retry(messages: list) -> str:
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=500
    )
    return response.choices[0].message.content
```

---

## 77. Production Considerations

**Token budget enforcement:**
Every production LLM endpoint should enforce a maximum token budget — both for input (prevent prompt injection attacks that try to exhaust context) and output (prevent runaway generation).

```python
MAX_INPUT_TOKENS = 8_000
MAX_OUTPUT_TOKENS = 1_000

async def safe_llm_call(messages: list) -> str:
    input_tokens = count_tokens(messages)
    
    if input_tokens > MAX_INPUT_TOKENS:
        raise ValueError(f"Input too large: {input_tokens} tokens (max {MAX_INPUT_TOKENS})")
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=MAX_OUTPUT_TOKENS
    )
    return response.choices[0].message.content
```

**Cost alerting:**
Set daily cost thresholds and alert when approached:
```python
# Track cumulative daily cost
daily_spend = sum(call.estimated_cost for call in today_calls)
if daily_spend > DAILY_BUDGET_ALERT_THRESHOLD:
    send_alert(f"LLM daily spend ${daily_spend:.2f} approaching budget")
```

**Latency SLAs:**
Track p50, p95, p99 latency. Output token count is the primary driver — a response that generates 2,000 tokens takes ~10x longer than one that generates 200 tokens for the same model.

**Graceful degradation:**
If the primary model is unavailable or rate-limited, fall back to a faster/cheaper model with a lower quality threshold — something is always better than nothing.

```python
async def llm_with_fallback(messages):
    try:
        return await call_model("gpt-4o", messages)
    except (RateLimitError, TimeoutError):
        return await call_model("gpt-4o-mini", messages)  # fallback
```

---

## 78. Common Misconceptions

**"The context window is the model's memory"**
No. The context window is the current working space, not memory. Memory implies persistence. The context window is wiped after every API call.

**"Larger context window = better model"**
Not necessarily. A 1M context window is useless if the model ignores content in the middle. Quality of attention across the context matters more than the size.

**"I can just send everything to the LLM and it'll figure out what's relevant"**
Sending irrelevant content hurts quality, increases cost, and wastes context space. Curate what you send.

**"Input and output tokens cost the same"**
Output tokens are 4–5x more expensive and take proportionally longer. Optimise both but watch output more carefully.

**"Prompt caching is automatic"**
Caching must be explicitly enabled (Anthropic requires `cache_control` parameter). It also requires the cached prefix to be identical in every call — even a single character change invalidates the cache.

**"More context always gives better RAG answers"**
False. The optimal number of retrieved chunks is usually 3–7, not 20+. More chunks introduce noise and the Lost in the Middle problem.

**"Token count is consistent across models"**
Different tokenizers produce different token counts for the same text. "Hello, how are you?" is 6 tokens in GPT-4 tiktoken but may be more or fewer in other models.

---

## 79. Interview Questions & Answers

### On Tokens & Tokenization

**Q: What is a token and why does tokenization exist?**
A token is the atomic unit of text processing in an LLM — a chunk of text that the model's vocabulary recognises as a single unit. Tokenization exists because neural networks operate on numbers, not text. Tokenization converts text to integer IDs which are then mapped to embedding vectors the model can process mathematically.

**Q: Why is subword tokenization preferred over word-level or character-level?**
Subword tokenization is the sweet spot: it keeps vocabulary manageable (30K–100K entries vs millions for word-level) while avoiding the OOV problem (unlike word-level) and keeping sequences short (unlike character-level where every character is a separate token). It also allows the model to generalise — "un-" appears as a prefix in thousands of words, and the model learns its meaning from shared subword tokens.

**Q: What are special tokens and why do they matter in production?**
Special tokens are reserved vocabulary entries that carry structural control signals — `<BOS>` marks sequence start, `<EOS>` tells the model to stop generating, `<PAD>` aligns batch sequences. In production, each model family has its own chat template using different special tokens. Using the wrong template silently degrades output quality because the model cannot correctly identify where the system prompt ends and user input begins.

### On Context Windows

**Q: Why is the context window limited in LLMs?**
Self-attention computes relationships between every pair of tokens — an O(n²) operation. Doubling the context length quadruples the computation and memory required. At 128K tokens, attention requires storing KV matrices for all 128K positions across all transformer layers — hundreds of GB of GPU memory per request. Context limits exist because there are physical limits to GPU memory and acceptable latency.

**Q: What is the "Lost in the Middle" problem and how do you mitigate it?**
LLMs demonstrably pay less attention to information placed in the middle of a long context compared to the beginning or end — forming a U-shaped attention curve. In RAG systems, this means the most relevant retrieved chunk may be ignored if placed in the middle. Mitigations include: placing most relevant chunks at the edges of context, reducing the total number of chunks (fewer chunks = less middle), using cross-encoder reranking to select only truly relevant content, and using structured XML tags to explicitly signal important content to the model.

**Q: What's the difference between context window and model memory?**
The context window is working memory — active for one API call only, wiped afterward. The model has no persistent memory between calls. "Memory" in AI applications is always implemented externally (vector database, Redis, relational DB) and injected into the context at the start of each call by the application layer.

### On Costs

**Q: Why do output tokens cost more than input tokens?**
Input tokens are processed in a single parallel forward pass — all tokens processed simultaneously. Output tokens are generated sequentially — each token requires a complete separate forward pass through the model. To generate 500 output tokens, the model runs 500 forward passes. This sequential compute cost is why output tokens are typically 4–5x more expensive than input tokens.

**Q: How would you reduce LLM API costs in production by 50%?**
Multiple approaches combined: switch from GPT-4o to GPT-4o-mini for tasks where quality is comparable (17x cost reduction), implement prompt caching for static system prompts (90% savings on repeated prefix), cache responses for identical queries in Redis, trim conversation history aggressively using summarisation, reduce RAG chunk count from 10 to 3–5 per query, use batch API for non-real-time workloads (50% discount), and set appropriate `max_tokens` caps per task type.

**Q: How do you estimate the monthly cost of an LLM feature before launch?**
Profile a representative sample of requests to measure average input and output tokens. Apply the formula: `monthly_cost = calls_per_day × 30 × ((avg_input_tokens/1M × input_price) + (avg_output_tokens/1M × output_price))`. Add a 30–50% buffer for variance and growth. Monitor actual usage post-launch against the estimate and set automated cost alerts at 80% of budget threshold.

**Q: What is prompt caching and when should you use it?**
Prompt caching stores the computed KV matrices for a repeated prefix of the prompt, so subsequent calls with the same prefix pay only a fraction of the normal input cost (90% cheaper for cache hits on Anthropic). Use it whenever you have a large, static system prompt or context that is identical across many calls — document analysis systems, large few-shot example sets, and tool definitions are ideal candidates. The cached prefix must be byte-for-byte identical across calls to get a cache hit.
