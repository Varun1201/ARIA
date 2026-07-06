"""
ARIA Evaluation Dataset
25 Q&A pairs with verified answers extracted directly from ingested papers.
Used to validate whether ARIA's faithfulness scores correlate with actual correctness.
"""

EVAL_DATASET = [
    # ── RAG Papers ────────────────────────────────────────────────────────────
    {
        "id": "rag_001",
        "question": "What does RAG stand for and what problem does it solve?",
        "expected_answer": "RAG stands for Retrieval-Augmented Generation. It solves the problem of knowledge-intensive NLP tasks by combining a parametric memory (the language model) with a non-parametric memory (a retrieval system over external documents), allowing models to access up-to-date information without retraining.",
        "source_paper": "2005.11401",
        "topic": "rag",
        "difficulty": "easy",
    },
    {
        "id": "rag_002",
        "question": "What are the two components of the RAG model architecture?",
        "expected_answer": "RAG combines a pre-trained seq2seq model (the generator) with a dense vector index of Wikipedia accessed with a pre-trained neural retriever. The retriever provides latent documents conditioned on the input, and the generator produces the output conditioned on those documents.",
        "source_paper": "2005.11401",
        "topic": "rag",
        "difficulty": "medium",
    },
    {
        "id": "rag_003",
        "question": "How does Dense Passage Retrieval encode questions and passages?",
        "expected_answer": "DPR uses two independent BERT encoders — one for questions and one for passages — to encode them into dense vector representations. Retrieval is performed by finding passages whose vectors have the highest dot product similarity with the question vector.",
        "source_paper": "2004.04906",
        "topic": "rag",
        "difficulty": "medium",
    },
    {
        "id": "rag_004",
        "question": "What is Corrective RAG and what problem does it address?",
        "expected_answer": "Corrective RAG (CRAG) addresses the problem of inaccurate or irrelevant retrieved documents corrupting the generation. It introduces a lightweight retrieval evaluator that assesses the quality of retrieved documents and triggers corrective actions such as web search when retrieval quality is poor.",
        "source_paper": "2401.15884",
        "topic": "rag",
        "difficulty": "medium",
    },
    {
        "id": "rag_005",
        "question": "What are the three paradigms of RAG described in the RAG survey?",
        "expected_answer": "The RAG survey describes three paradigms: Naive RAG (the basic retrieve-then-read pipeline), Advanced RAG (which adds pre-retrieval and post-retrieval processing steps to improve quality), and Modular RAG (which restructures the pipeline into independent interchangeable modules for greater flexibility).",
        "source_paper": "2312.10997",
        "topic": "rag",
        "difficulty": "hard",
    },
    {
        "id": "rag_006",
        "question": "What evaluation framework does the RAG survey introduce?",
        "expected_answer": "The RAG survey introduces an up-to-date evaluation framework covering three quality scores — context relevance, answer faithfulness, and answer relevance — along with four key abilities: noise robustness, negative rejection, information integration, and counterfactual robustness.",
        "source_paper": "2312.10997",
        "topic": "rag",
        "difficulty": "hard",
    },
    {
        "id": "rag_007",
        "question": "What is Graph RAG and how does it differ from standard RAG?",
        "expected_answer": "Graph RAG uses a knowledge graph constructed from source documents rather than raw text chunks. It performs community detection on the graph and generates community summaries, enabling query-focused summarization over entire document corpora — something standard vector-based RAG cannot do effectively.",
        "source_paper": "2404.16130",
        "topic": "rag",
        "difficulty": "hard",
    },

    # ── LLM Foundation Papers ─────────────────────────────────────────────────
    {
        "id": "llm_001",
        "question": "What is the core innovation of the Transformer architecture in Attention Is All You Need?",
        "expected_answer": "The Transformer replaces recurrent and convolutional layers entirely with a self-attention mechanism. The key innovation is multi-head attention, which allows the model to jointly attend to information from different representation subspaces at different positions, enabling parallelization and better long-range dependency modeling.",
        "source_paper": "1706.03762",
        "topic": "llm_foundation",
        "difficulty": "easy",
    },
    {
        "id": "llm_002",
        "question": "What is the model size of GPT-3 and how many parameters does it have?",
        "expected_answer": "GPT-3 has 175 billion parameters, making it the largest dense language model at the time of publication. It was trained on 300 billion tokens of text data.",
        "source_paper": "2005.14165",
        "topic": "llm_foundation",
        "difficulty": "easy",
    },
    {
        "id": "llm_003",
        "question": "What is the context length supported by Mistral 7B?",
        "expected_answer": "Mistral 7B supports a context length of 8192 tokens, achieved through sliding window attention (SWA) which allows attending to tokens beyond the window size through stacked attention layers.",
        "source_paper": "2310.06825",
        "topic": "llm_foundation",
        "difficulty": "medium",
    },
    {
        "id": "llm_004",
        "question": "What technique does Mistral 7B use to improve inference speed?",
        "expected_answer": "Mistral 7B uses grouped-query attention (GQA) to speed up inference and reduce memory usage during decoding, alongside sliding window attention (SWA) for efficient handling of long sequences.",
        "source_paper": "2310.06825",
        "topic": "llm_foundation",
        "difficulty": "medium",
    },
    {
        "id": "llm_005",
        "question": "What is RLHF and how is it used in InstructGPT?",
        "expected_answer": "RLHF stands for Reinforcement Learning from Human Feedback. In InstructGPT, it is used in three steps: supervised fine-tuning on human demonstrations, training a reward model on human preference comparisons, and optimizing the language model against the reward model using PPO (Proximal Policy Optimization).",
        "source_paper": "2203.02155",
        "topic": "llm_foundation",
        "difficulty": "medium",
    },
    {
        "id": "llm_006",
        "question": "How many parameters does LLaMA 2 have in its largest variant?",
        "expected_answer": "LLaMA 2's largest variant has 70 billion parameters. The model family includes 7B, 13B, and 70B parameter versions, all trained on 2 trillion tokens of publicly available data.",
        "source_paper": "2307.09288",
        "topic": "llm_foundation",
        "difficulty": "easy",
    },
    {
        "id": "llm_007",
        "question": "What context length does LLaMA 2 support compared to LLaMA 1?",
        "expected_answer": "LLaMA 2 doubles the context length of LLaMA 1, supporting 4096 tokens compared to LLaMA 1's 2048 tokens.",
        "source_paper": "2307.09288",
        "topic": "llm_foundation",
        "difficulty": "medium",
    },

    # ── Agents Papers ─────────────────────────────────────────────────────────
    {
        "id": "agent_001",
        "question": "What does ReAct stand for and what is its core idea?",
        "expected_answer": "ReAct stands for Reasoning and Acting. Its core idea is to interleave reasoning traces (chain-of-thought) and task-specific actions in an LLM, allowing the model to reason about what to do, take an action, observe the result, and continue reasoning — creating a synergy between reasoning and acting.",
        "source_paper": "2210.03629",
        "topic": "agents",
        "difficulty": "easy",
    },
    {
        "id": "agent_002",
        "question": "What is Toolformer and how does it learn to use tools?",
        "expected_answer": "Toolformer is a model that learns to use external tools (calculator, search engine, calendar, etc.) through self-supervised learning. It is trained on data where the model itself decides which API calls to make, when to make them, and how to incorporate the results — without requiring large amounts of human annotation.",
        "source_paper": "2302.04761",
        "topic": "agents",
        "difficulty": "medium",
    },
    {
        "id": "agent_003",
        "question": "What advantage does ReAct have over chain-of-thought reasoning alone?",
        "expected_answer": "ReAct has the advantage of grounding reasoning in external information through actions like search, which reduces hallucination and allows the model to update its reasoning based on real observations. Chain-of-thought alone can generate plausible but factually incorrect reasoning chains without any external grounding.",
        "source_paper": "2210.03629",
        "topic": "agents",
        "difficulty": "hard",
    },

    # ── Evaluation Papers ─────────────────────────────────────────────────────
    {
        "id": "eval_001",
        "question": "What is TruthfulQA and what does it measure?",
        "expected_answer": "TruthfulQA is a benchmark that measures whether language models generate truthful answers. It consists of 817 questions designed to elicit imitative falsehoods — answers that models learn from training data even though they are false. It tests whether models can avoid generating popular misconceptions.",
        "source_paper": "2109.07958",
        "topic": "evaluation",
        "difficulty": "easy",
    },
    {
        "id": "eval_002",
        "question": "What does FActScore measure and how does it work?",
        "expected_answer": "FActScore (Factual precision in Atomic Score) measures the factual precision of long-form text generation. It works by breaking generated text into atomic facts, then verifying each atomic fact independently against a knowledge source, computing the percentage of atomic facts that are supported.",
        "source_paper": "2307.03025",
        "topic": "evaluation",
        "difficulty": "medium",
    },
    {
        "id": "eval_003",
        "question": "What are the four metrics RAGAS uses to evaluate RAG pipelines?",
        "expected_answer": "RAGAS uses four metrics: faithfulness (whether the answer is grounded in the retrieved context), answer relevancy (how relevant the answer is to the question), context precision (whether the retrieved context is relevant), and context recall (whether the retrieved context covers the ground truth answer).",
        "source_paper": "2309.01431",
        "topic": "evaluation",
        "difficulty": "hard",
    },
    {
        "id": "eval_004",
        "question": "What is the main finding of the hallucination survey regarding LLM hallucinations?",
        "expected_answer": "The hallucination survey finds that LLM hallucinations stem from multiple sources including the pre-training data (noise and misinformation), the training process (exposure bias, imperfect decoding), and the inference stage (factual knowledge gaps). It categorizes hallucinations into intrinsic (contradicting source) and extrinsic (unverifiable) types.",
        "source_paper": "2309.07864",
        "topic": "evaluation",
        "difficulty": "hard",
    },

    # ── Cross-topic questions ─────────────────────────────────────────────────
    {
        "id": "cross_001",
        "question": "What is the difference between parametric and non-parametric memory in language models?",
        "expected_answer": "Parametric memory refers to knowledge stored in the model's weights during training — implicit and fixed after training. Non-parametric memory refers to external knowledge stored in explicit data structures like databases or document indices that can be updated without retraining the model. RAG combines both.",
        "source_paper": "2005.11401",
        "topic": "rag",
        "difficulty": "medium",
    },
    {
        "id": "cross_002",
        "question": "How does sliding window attention in Mistral differ from full self-attention?",
        "expected_answer": "In full self-attention every token attends to all previous tokens, making memory and computation grow quadratically with sequence length. Sliding window attention restricts each token to attend only to the W nearest tokens, making it linear in sequence length, while still capturing long-range dependencies through stacked layers.",
        "source_paper": "2310.06825",
        "topic": "llm_foundation",
        "difficulty": "hard",
    },
    {
        "id": "cross_003",
        "question": "What is the tripartite foundation of RAG frameworks?",
        "expected_answer": "According to the RAG survey, the tripartite foundation of RAG frameworks consists of three core components: retrieval (finding relevant information from external knowledge sources), augmentation (incorporating the retrieved information into the generation process), and generation (producing the final answer using both the model's parametric knowledge and the retrieved context).",
        "source_paper": "2312.10997",
        "topic": "rag",
        "difficulty": "medium",
    },
]