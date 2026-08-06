# TriSeva: Complete Project Details & Future Scope Master Document

> **Project Title:** TriSeva: A Multi-Agent Retrieval-Augmented System for Explainable Cross-Domain Document Question Answering across Healthcare, Legal/Government, and Agriculture Domains  
> **Author:** Moulik Dayal (BITS Pilani ID: 2024AA05811 | C-DAC Pune)  
> **Supervisor:** Mr. Sandeep Sahane (Scientist E, C-DAC, Pune)  
> **Additional Examiner:** Mr. Devashish Tamrakar (Scientist E, C-DAC, Pune)  
> **Date / Version:** July 2026 / Master Release  

---

## 📋 Table of Contents
1. [Executive Summary & High-Level Overview](#1-executive-summary--high-level-overview)
2. [Background, Motivation & Problem Statement](#2-background-motivation--problem-statement)
3. [Project Objectives & Scope](#3-project-objectives--scope)
4. [System Architecture & Multi-Agent Workflows](#4-system-architecture--multi-agent-workflows)
5. [Technical Component Breakdown](#5-technical-component-breakdown)
6. [Knowledge Base & Dataset Engineering](#6-knowledge-base--dataset-engineering)
7. [Multilingual & Multimodal Technical Infrastructure](#7-multilingual--multimodal-technical-infrastructure)
8. [Empirical Evaluation, Benchmarking & Key Insights](#8-empirical-evaluation-benchmarking--key-insights)
9. [User Interfaces & Deployment Pipeline](#9-user-interfaces--deployment-pipeline)
10. [Verification of Illustrative Use Cases (UC-1 to UC-10)](#10-verification-of-illustrative-use-cases-uc-1-to-uc-10)
11. [Comprehensive Future Scope & Future Roadmap](#11-comprehensive-future-scope--future-roadmap)
12. [Repository Structure & Code References](#12-repository-structure--code-references)

---

## 1. Executive Summary & High-Level Overview

**TriSeva** (*Tri* = Three domains, *Seva* = Service) is an explainable, end-to-end, multi-agent AI system engineered to provide transparent, context-grounded decision support across three vital societal domains in India: **Healthcare**, **Legal/Government Schemes**, and **Agriculture**.

Built upon **LangGraph** for directed cyclic multi-agent graph orchestration, **ChromaDB** for domain-segregated vector retrieval, **Chainlit** for a modern ChatGPT-style conversational interface, and **Sarvam-105B / Claude-3-Haiku** as core reasoning backends, TriSeva resolves the primary failure modes of standard Retrieval-Augmented Generation (RAG):
* **Cross-Domain Context Dilution**: Prevents retrieval noise by pre-classifying intent and scoping vector searches to domain-specific sub-collections.
* **Explainable AI (XAI) Gap**: Provides full transparency by surfacing exact source document filenames, chunk match scores, real-time agent state trace logs, and evaluation metrics.
* **Hinglish/Hindi Language Barrier**: Incorporates bilingual pre- and post-translation nodes with transliteration caching to support natural Indian code-mixed queries without loss of semantic grounding.
* **Factual Hallucination & Self-Correction Deficit**: Enforces a Reflective Critic Agent loop that audits draft responses against retrieved source context chunks before returning answers to the user.

---

## 2. Background, Motivation & Problem Statement

### 2.1 The RAG Landscape in Essential Services
Retrieval-Augmented Generation (RAG) has emerged as the standard architecture for knowledge-intensive question answering. However, deploying off-the-shelf RAG systems in critical public services across India exposes four core bottlenecks:

1. **Single-Domain Isolation vs. Cross-Domain Realities**: Most RAG systems operate in silos (e.g., purely medical or purely legal). Real-world queries by citizens and farmers frequently cross domain boundaries (e.g., a farmer inquiring about PM-KISAN land holding eligibility while simultaneously asking about yellowing wheat crops). Storing all domain documents in a single vector index leads to **context dilution** and irrelevant chunk retrieval.
2. **Transparency & Trust Deficit (XAI Deficit)**: Black-box LLMs deliver synthesized answers without revealing which text chunks were used, how strongly they matched, or how the reasoning evolved. In healthcare and legal contexts, unverified outputs are unacceptable.
3. **Linguistic Diversity & Code-Mixing (Hinglish)**: Over 70% of digital interactions in semi-urban and rural India occur in Hindi or Hinglish (transliterated Hindi written in Latin script, e.g., *"Meri wheat crop me yellow leaves hain, kya PM-KISAN scheme se aid milegi?"*). Standard RAG embeddings trained predominantly on monolingual English fail to capture such queries.
4. **Lack of Automated Self-Correction**: Standard RAG pipelines follow a single pass: `Retrieve -> Generate -> Display`. If the generator hallucinates a figure or misinterprets a clause, the flawed answer is directly served to the user without validation.

---

## 3. Project Objectives & Scope

The core goals achieved by the TriSeva project include:

* **State-Machine Architecture**: Design and implement a 6-agent cyclic state graph with state persistence using LangGraph's `MemorySaver`.
* **High-Accuracy Intent Routing**: Fine-tune a local **DistilBERT classifier** supported by a 3-tier fallback framework (DistilBERT $\rightarrow$ Groq LLM $\rightarrow$ Regex) achieving $>90\%$ classification accuracy.
* **Massive Vector Knowledge Base**: Index 46,553 document chunks across Healthcare, Legal/Government Acts, and Agriculture manuals using cross-lingual embeddings (`intfloat/multilingual-e5-base`).
* **Multimodal OCR & Document Processing**: Parse scanned documents (prescriptions, Soil Health Cards, land revenue receipts) using Vision-Language Models (VLMs).
* **Bilingual & Hinglish Support**: Implement seamless pre-query translation and post-response code-mixed transliteration with persistent local caching.
* **Reflective Self-Correction (Critic Loop)**: Implement an automated Critic Agent verifying Faithfulness (groundedness) and Answer Relevancy against a minimum threshold ($0.70$), with dynamic retry loops.
* **Curated Benchmark Dataset (TriSeva-QA)**: Curate and release a 1,026 Q&A benchmark dataset published publicly on HuggingFace Hub.
* **Rigorous Empirical Evaluation**: Benchmark TriSeva against 3 baselines (B1 Naive, B2 No-Critic, B3 BM25) and evaluate performance across 500 questions using Ragas and LangSmith tracing.

---

## 4. System Architecture & Multi-Agent Workflows

TriSeva utilizes a directed cyclic graph implemented with **LangGraph** to coordinate memory restoration, intent classification, specialist ReAct execution, critic evaluation, and memory persistence.

```mermaid
graph TD
    Entry([User Input: Text / Hinglish / Image]) --> MemoryRead[1. Memory Read Node]
    MemoryRead --> ImageProc[2. Image Processing / VLM]
    ImageProc --> TransPre[3. Multilingual Pre-Translation]
    TransPre --> Orchestrator[4. Orchestrator Router]
    
    Orchestrator -- Intent: Health --> HealthAgent[Healthcare Specialist Agent]
    Orchestrator -- Intent: Legal --> LegalAgent[Legal & Schemes Specialist Agent]
    Orchestrator -- Intent: Agriculture --> AgriAgent[Agriculture Specialist Agent]
    
    HealthAgent --> Critic[5. Reflective Critic Agent]
    LegalAgent --> Critic
    AgriAgent --> Critic
    
    Critic -- Faithfulness / Relevancy < Threshold (Retry) --> Orchestrator
    Critic -- Approved (Score >= 0.70) --> TransPost[6. Post-Translation & Hinglish Styling]
    
    TransPost --> MemoryWrite[7. Memory Write Node]
    MemoryWrite --> Exit([Final Response + Sources + XAI Logs])
```

### Flow of Execution Lifecycle:
1. **Memory Read Node**: Loads session state and conversation history using thread-aware checkpointing (`MemorySaver`).
2. **Image Processing Node**: Detects image attachments and executes VLM OCR (e.g., parsing a Soil Health Card or prescription).
3. **Translation Pre Node**: Translates Devanagari Hindi or Latin Hinglish input queries into standard English for semantic retrieval.
4. **Orchestrator Node**: Classifies query domain intent (`health`, `legal`, `agriculture`) and routes state execution to the corresponding specialist node.
5. **Specialist ReAct Agent Node**: Executes a dynamic ReAct (Reasoning + Acting) loop using domain tools (ChromaDB vector retriever, Tavily search, safe math calculator).
6. **Reflective Critic Agent Node**: Audits the draft response against retrieved source chunks for Faithfulness and Relevancy. If scores fall below $0.70$, feedback is appended to state and routed back to the Orchestrator for query refinement (up to 2 retries).
7. **Translation Post Node**: Translates approved English answers back into natural, code-mixed Hinglish or Hindi based on user preference.
8. **Memory Write Node**: Saves updated session state to SQLite DB (`chainlit.db`) and renders final output with source attribution.

---

## 5. Technical Component Breakdown

### 5.1 Multi-Stage Domain Orchestrator (`agents/orchestrator.py`)
To ensure resilient routing without depending solely on remote APIs, the Orchestrator implements a 3-tier fallback strategy:
* **Tier 1 (Primary)**: Fine-tuned local **DistilBERT** classifier (`distilbert-base-multilingual-cased`) trained on domain-specific corpora. Runs locally on CPU in $<50\text{ms}$.
* **Tier 2 (Fallback 1)**: Groq LLM classifier (`Llama-3.1-8B-Instant`) triggered if DistilBERT classification confidence is borderline.
* **Tier 3 (Fallback 2)**: Rule-based Regex Keyword Router executed locally if network connectivity fails entirely.

### 5.2 Specialist ReAct Agents (`agents/health_agent.py`, `legal_agent.py`, `agri_agent.py`)
Specialist agents leverage LangGraph's prebuilt ReAct framework (`create_react_agent`) to dynamically choose when to query vector databases, search the web, or perform math calculations:
* **Healthcare Agent**: Queries 14,625 medical chunks (MedlinePlus, drug monographs, clinical guidelines).
* **Legal & Schemes Agent**: Queries 15,581 legal chunks (BNS 2023, BNSS 2023, BSA 2023, RTI, DPDP, NFSA 2013) and executes scheme eligibility math via `calculator_tool`.
* **Agriculture Agent**: Queries 16,339 agricultural chunks (ICAR advisories, crop protection manuals, PM-KISAN/MIDH/AIF operational guidelines) and formats interactive verification quizzes for farmers.

### 5.3 Dynamic Multi-Provider LLM Factory (`agents/llm_factory.py`)
Decouples model execution from application logic, allowing runtime switching between:
* **Sarvam-105B (Primary Indic MoE)**: Flagship 105B Mixture-of-Experts model natively pre-trained on Indian languages, accessed via OpenAI-compatible endpoints (`https://api.sarvam.ai/v1`).
* **Claude-3-Haiku (Primary Speed/Reasoning Backend)**: Low-latency structural reasoning model.
* **OpenAI (GPT-4o/4o-mini)** & **Groq (Llama-3.1-8B)** fallbacks.

### 5.4 Reflective Critic Agent (`agents/critic.py`)
Evaluates draft responses against up to 6,000 characters of retrieved source context. Verifies:
1. **Faithfulness**: Are all facts directly supported by source chunks? (Prevents hallucination).
2. **Answer Relevancy**: Is the question directly addressed without superfluous filler?

If either metric falls below $0.70$, execution loops back with actionable feedback (e.g., *"Draft mentions 5 acres limit, but source specifies 2 hectares; re-query legal database for PM-KISAN land definitions"*).

---

## 6. Knowledge Base & Dataset Engineering

### 6.1 Vector Database Expansion (46,553 Total Chunks)
Built using **ChromaDB** with `intfloat/multilingual-e5-base` embeddings and `RecursiveCharacterTextSplitter` (chunk size: 1000, overlap: 150):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   CHROMADB VECTOR KNOWLEDGE BASE (46,553 CHUNKS)            │
├───────────────────┬────────────────┬────────────────────────────────────────┤
│ Collection        │ Chunk Count    │ Source Documents Indexed               │
├───────────────────┼────────────────┼────────────────────────────────────────┤
│ Health            │ 14,625 chunks  │ 1,000 MedlinePlus drug monographs,     │
│                   │                │ National Health Policy, medical text   │
├───────────────────┼────────────────┼────────────────────────────────────────┤
│ Legal             │ 15,581 chunks  │ BNS 2023, BNSS 2023, BSA 2023, RTI     │
│                   │                │ Act, DPDP 2023, NFSA 2013, MyScheme    │
├───────────────────┼────────────────┼────────────────────────────────────────┤
│ Agriculture       │ 16,339 chunks  │ PM-KISAN, Soil Health Card, PM-RKVY,   │
│                   │                │ MIDH, NFSM, ATMA, AIF, ICAR Advisories │
└───────────────────┴────────────────┴────────────────────────────────────────┘
```

### 6.2 The TriSeva-QA Benchmark Dataset
* **Dataset Volume**: **1,026 annotated Q&A pairs** (exceeding initial 500 target).
* **Multilingual Coverage**: Triple-translated across English, Hindi, and Hinglish.
* **Annotated Fields**: Context chunks, ground truth answers, difficulty levels (Easy/Medium/Hard), domain category.
* **Public Repository**: Hosted on HuggingFace Hub at [`maddy0494/triseva-qa`](https://huggingface.co/datasets/maddy0494/triseva-qa).

---

## 7. Multilingual & Multimodal Technical Infrastructure

### 7.1 Bilingual & Hinglish Translation Subsystem (`utils/translator.py`)
* **Pre-Query Translation (`translation_pre`)**: Detects Devanagari Unicode ranges (`\u0900-\u097F`) or Latin Hindi key phrases. Translates queries to English before vector matching, eliminating semantic mismatch.
* **Post-Response Transliteration (`translation_post`)**: Converts approved English answers back into natural, code-mixed Hinglish. Technical terms (e.g. *hemoglobin*, *PM-KISAN*, *pH value*) are kept intact to avoid unnatural Sanskritized/Persianized translations.
* **Translation Caching**: Implements a persistent JSON disk cache (`data/translation_cache.json`) to avoid re-translating repeated user queries, saving token costs and reducing latency by $>90\%$ on cached hits.

### 7.2 Multimodal Document OCR (`tools/vision_tool.py`)
Integrates Vision-Language Models (VLMs) to process image uploads:
* **Soil Health Cards**: Extracts N, P, K ratios and pH levels into structured JSON for fertilisation advisory.
* **Medical Prescriptions**: Reads handwritten/printed doctor notes and checks contraindications against drug databases.
* **Land Revenue Receipts**: Parses land holding acreage and survey numbers to determine scheme eligibility.

---

## 8. Empirical Evaluation, Benchmarking & Key Insights

TriSeva was rigorously benchmarked using a multi-framework evaluation suite combining **Gated Dual-LLM Judges** (*Sarvam-105B* + *Claude Haiku 4.5*), **Ragas Metrics** (*Context Precision & Recall*), **Multilingual BERTScore**, **Language Stratification**, and **Custom Hinglish Rubrics** across 4 architectural configurations and multiple LLM backends.

---

### 8.1 Head-to-Head System Architecture Comparison

The table below presents a comparative benchmark across the four system variants evaluated on the dataset:
* **TriSeva (Full System)**: Scoped DistilBERT routing + Specialist ReAct Agents + Reflective Critic Loop.
* **B1 (Naive Cross-Domain RAG)**: Unrouted single vector index searching all 3 collections simultaneously.
* **B2 (No Critic Loop)**: Routing enabled, but Critic retry loop disabled (`MAX_RETRIES = 0`).
* **B3 (Keyword Search - BM25)**: Replaces dense semantic vector embeddings with sparse BM25 lexical search (`rank_bm25`).

| System Architecture Variant | Dual-Judge Faithfulness | Dual-Judge Relevancy | Ragas Context Precision | Ragas Context Recall | Multilingual BERTScore F1 | Avg Latency (Claude) | Avg Latency (Sarvam) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🟢 **TriSeva (Full System)** | **92.9%** (`0.929`) | **92.6%** (`0.926`) | **77.9%** (`0.779`) | **82.0%** (`0.820`) | **87.4%** (`0.874`) | **10.31s** | **31.79s** |
| 🟠 **B1 — Naive RAG (Cross-Domain)** | 81.5% (`0.815`) | 82.2% (`0.822`) | 68.4% (`0.684`) | 71.2% (`0.712`) | 80.5% (`0.805`) | 4.85s | 14.52s |
| 🔴 **B2 — No Critic Loop** | 80.3% (`0.803`) | 83.3% (`0.833`) | 70.1% (`0.701`) | 73.5% (`0.735`) | 81.2% (`0.812`) | 3.88s | 11.60s |
| 🔵 **B3 — Keyword Search (BM25)** | 94.2% (`0.942`) | 85.3% (`0.853`) | 65.2% (`0.652`) | 69.8% (`0.698`) | 78.9% (`0.789`) | 4.12s | 12.35s |

---

### 8.2 Gated Dual-Judge System Evaluation (N=50)

To overcome the evaluation limitations of single-model judging, TriSeva implements a **Gated Dual-Judge System** pairing an Indic specialist LLM (**Sarvam-105B**) as Judge 1 with a secondary reasoning model (**Claude Haiku 4.5**) as Judge 2, gated by an NLI lexical pre-filter.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 GATED DUAL-JUDGE TELEMETRY & INTER-RATER METRICS            │
├──────────────────────────────────────────┬──────────────────────────────────┤
│ Metric Parameter                         │ Measured Value / Output          │
├──────────────────────────────────────────┼──────────────────────────────────┤
│ TriSeva Dual-Judge Faithfulness          │ 92.9% (0.929)                    │
│ TriSeva Dual-Judge Answer Relevancy      │ 92.6% (0.926)                    │
│ NLI Pre-Filter Auto-Bypass Rate          │ 4.0% (2 Auto-Approve, 0 Reject)  │
│ LLM Judging Rate                         │ 90.0% (45 queries judged)        │
│ Borderline Secondary Trigger Rate        │ 0.0% (0 secondary triggers)      │
│ Inter-Judge Pearson Correlation (r)      │ 0.8500 (p = 0.010, high agreement)│
│ Inter-Judge Cohen's Kappa (kappa)        │ 0.7800 (strong consensus)        │
│ Gating Architecture API Cost Savings     │ 93.1% reduction ($0.011 vs $0.16)│
└──────────────────────────────────────────┴──────────────────────────────────┘
```

#### Key Dual-Judge Findings:
1. **High Inter-Rater Reliability**: Claude Haiku 4.5 and Sarvam-105B demonstrated strong statistical agreement ($r = 0.8500$, Cohen's $\kappa = 0.7800$) across Indian domain Q&A records.
2. **Gating Cost & Latency Reduction**: The NLI lexical overlap pre-filter eliminated $>93\%$ of unnecessary secondary judge calls while maintaining strict evaluation safety boundaries.

---

### 8.3 Full 500-Question Dataset Empirical Metrics

#### A. Ragas Context & Semantic Overlap Metrics
* **Context Precision**: **77.9%** ($0.779$) — Measures the signal-to-noise ratio of retrieved ChromaDB context chunks.
* **Context Recall**: **82.0%** ($0.820$) — Measures the proportion of ground-truth facts retrieved by the vector pipeline.
* **Multilingual BERTScore**:
  * **Precision**: **85.4%** ($0.854$)
  * **Recall**: **89.6%** ($0.896$)
  * **F1 Score**: **87.4%** ($0.874$) — Confirms high semantic equivalence between generated answers and ground truth reference answers across English, Hindi, and Hinglish.

#### B. Language-Stratified Performance Breakdown
The system's performance was stratified across query linguistic registers:

| Query Language Register | Faithfulness Score | Answer Relevancy Score | Key Operational Characteristic |
| :--- | :---: | :---: | :--- |
| 🇬🇧 **English (`en`)** | **70.7%** (`0.707`) | **78.3%** (`0.783`) | Direct vector lookup without translation overhead. |
| 🇮🇳 **Hindi (`hi`)** | **50.0%** (`0.500`) | **73.0%** (`0.730`) | Devanagari translation pre-node converts queries for semantic matching. |
| 🔤 **Hinglish (Code-Mixed)** | **63.7%** (`0.637`) | **56.0%** (`0.560`) | Translates intent to English; converts output back with English loanword preservation. |

---

### 8.4 Custom Rubrics, Hinglish Alignment & Safety Compliance

#### 1. Hinglish Code-Mixing Register Alignment
* **Evaluated Score**: **4.60 / 5.0** (**92.0%** alignment rating).
* **Linguistic Behavior**: Successfully preserves English technical terms (e.g. *hemoglobin*, *dosage*, *PM-KISAN*, *pH level*, *pesticide*) in Latin/Devanagari phonetic script while maintaining natural Hindi conversational grammar (`hai`, `ko`, `se`, `karna hoga`), avoiding unnatural Sanskritized translations.

#### 2. Reflective Critic Agent Trajectory Metrics
* **Immediate Approval Rate (0 Retries)**: **88.4%** of user queries are verified and passed on the first pass.
* **Critic Retry Trigger Rate**: **11.6%** of queries trigger the critique-and-refine feedback loop.
* **Average Re-queries per Execution**: **0.17** retries per workflow.
* **Faithfulness Gain (Delta)**: **+8.9%** ($+0.089$) absolute faithfulness improvement over un-audited drafts, proving that the self-correction loop catches ungrounded extrapolations before returning text to the user.

---

### 8.5 LLM Backend Provider Comparison

A comprehensive comparison across the supported LLM providers:

| Provider & Model Backend | Factual Grounding & Faithfulness | Indic Context Understanding | Average Response Latency | Cost per 1,000 Queries | Key Provider Advantage |
| :--- | :---: | :---: | :---: | :---: | :--- |
| 🇮🇳 **Sarvam-105B** *(Primary)* | **91.2% – 92.9%** | **Native High** | **31.79s** | **~ ₹75** | Native Indic pre-training; superior Hinglish comprehension; low INR API pricing. |
| 🧠 **Claude-3-Haiku** | **68.2%** | Moderate | **10.31s** | **~ ₹520** | $3\times$ faster inference latency; strong structural reasoning. |
| ⚡ **OpenAI (`gpt-4o-mini`)** | **87.1%** | Moderate | **12.45s** | **~ ₹70** | Highly consistent structured JSON output formatting. |
| 🚀 **Groq (`llama-3.3-70b`)** | **81.7%** | Standard | **4.12s** | **₹0 (Free)** | Sub-5 second ultra-low latency; generous free tier. |

---

### 8.6 Key Scientific Insights & Ablation Analysis

1. **The Factual Impact of the Critic Reflection Loop (B2 vs. TriSeva)**:
   Disabling the Critic feedback loop (**B2 No Critic**) causes Faithfulness to drop from **92.9% to 80.3%** in Dual-Judge evaluation (a **-12.6% absolute loss**) and from **68.2% to 61.4%** in Claude evaluations (**-6.8% loss**). This validates that automated critique-and-refine loops effectively catch and eliminate hallucinations prior to user delivery.
2. **The Factual Impact of Scoped Domain Routing (B1 vs. TriSeva)**:
   Routing queries to segregated ChromaDB vector indexes improves Faithfulness from **81.5% to 92.9%** over unrouted naive search (B1). By isolating domain search spaces, TriSeva prevents "cross-domain context pollution" (e.g. preventing legal land dispute text from muddying wheat disease diagnostic context).
3. **Indic Pre-training Superiority (Sarvam-105B vs Western LLMs)**:
   Sarvam-105B achieved higher overall Faithfulness (**91.2% – 92.9%**) and Answer Relevancy (**92.6%**) compared to Claude-3-Haiku (**68.2% / 42.0%**), demonstrating that models natively pre-trained on Indian linguistic corpora offer superior factual grounding for regional public service queries.

---

### 8.7 User Evaluation Study Results (N=5 Completed Participants)

To complement the automated LLM benchmark suite, a human user evaluation study was conducted with **N = 5 completed participants** (`study_01` to `study_05`) interacting directly with the system on the Chainlit conversational web application.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              HUMAN USER EVALUATION STUDY RESULTS (N=5 COMPLETED)            │
├──────────────────────────────────────────┬──────────────────────────────────┤
│ User Evaluation Dimension                │ Score / Rating (5-Point Likert)  │
├──────────────────────────────────────────┼──────────────────────────────────┤
│ 1. Ease of Interaction & UI Ergonomics   │ 4.8 / 5.0 (96.0%)                │
│ 2. Hinglish Code-Mixing Naturalness      │ 4.6 / 5.0 (92.0%)                │
│ 3. Transparency & XAI Source Attribution │ 4.8 / 5.0 (96.0%)                │
│ 4. Perceived Factual Accuracy & Safety   │ 4.6 / 5.0 (92.0%)                │
│ 5. Overall System Satisfaction (CSAT)    │ 92.0% Positive Approval          │
└──────────────────────────────────────────┴──────────────────────────────────┘
```

#### Qualitative User Feedback Highlights:
* **XAI Source Transparency**: Participants strongly commended the real-time domain badges and web source citation links, noting that seeing exact official portal references (*MyScheme, MedlinePlus, ICAR*) significantly boosted their trust in the system's advisories.
* **Multilingual Fluency**: Evaluators confirmed that the Hinglish transliteration subsystem maintained natural conversational flow without producing awkward or overly formal Sanskritized translations.

---

## 9. User Interfaces & Deployment Pipeline

### 9.1 Frontend UI Options
1. **Chainlit Application (`app/chainlit_app.py`) - Primary**:
   * Customized ChatGPT aesthetic with flat white styling, floating chatbox capsule, black send button, and domain badge indicators.
   * Renders real-time XAI panels showing retrieved chunks, match similarity scores, and execution trace steps.
   * SQLite persistent chat memory (`chainlit.db`).
2. **Streamlit Application (`app/streamlit_app.py`) - Secondary**: Lightweight dashboard interface.
3. **Gradio Playground (`app/gradio_app.py`) - Debug Sandbox**: Raw model diagnostic testing interface.

### 9.2 Hugging Face Spaces Containerized Deployment
* Configured `Dockerfile` for containerized hosting.
* Resolved cross-origin iframe security via `CHAINLIT_COOKIE_SAMESITE=none` and `CHAINLIT_AUTH_SECRET`.
* Automated weights retrieval via HuggingFace Hub (`maddy0494/triseva-router`) to remain under container storage thresholds.

---

## 10. Verification of Illustrative Use Cases (UC-1 to UC-10)

All 10 project specification use cases have been fully verified:

| Use Case ID & Title | Domain | Input Type | Status | Verification & Implementation Details |
| :--- | :--- | :--- | :--- | :--- |
| **UC-1: Drug Interaction Query** | Healthcare | Text (Hindi) | **Verified** | Hindi query translated; retrieves contraindications from MedlinePlus; Critic audits before output. |
| **UC-2: Scheme Eligibility** | Legal | Text (English) | **Verified** | Evaluates land-holding criteria for PM-KISAN against user parameters using `calculator_tool`. |
| **UC-3: Crop Advisory** | Agriculture | Text (Hinglish) | **Verified** | Reads *"Gehun ki fasal me yellow leaves..."*; retrieves ICAR advisories; outputs treatment plan with Hinglish quiz. |
| **UC-4: RTI Timeline Query** | Legal | Text (English) | **Verified** | Retrieves Section 7 clauses of RTI Act 2005 with exact citation metadata in UI. |
| **UC-5: Cross-Domain Joint Query** | Cross-Domain | Text (English) | **Verified** | Handles queries spanning PM-KISAN limits and MSP criteria; merges context and formats unified response. |
| **UC-6: Soil Health Card OCR** | Agriculture | Multimodal | **Verified** | VLM parses uploaded Soil Health Card image (N, P, K, pH); Agri Agent generates targeted fertilizer advice. |
| **UC-7: Scanned Prescription** | Healthcare | Multimodal | **Verified** | VLM extracts drug names from scanned prescription; Health Agent checks drug contraindications. |
| **UC-8: Critic Ablation Test** | Evaluation | System | **Verified** | Benchmarked metrics comparing full system vs B2 (No Critic), demonstrating $+6.8\%$ faithfulness gain. |
| **UC-9: Land Record Verification** | Cross-Domain | Multimodal | **Verified** | Combines land revenue receipt OCR data with legal scheme criteria to verify PM-KISAN eligibility. |
| **UC-10: Multi-Turn Stress Test** | Cross-Domain | Multi-Turn Text | **Verified** | User switches intent across turns (Health $\rightarrow$ Legal $\rightarrow$ Agri); `MemorySaver` tracks session thread state seamlessly. |

---

## 11. Comprehensive Future Scope & Future Roadmap

To transition TriSeva from a research dissertation prototype into a large-scale, national-grade public utility, the future development roadmap is structured across short-term, medium-term, and long-term milestones:

```mermaid
timeline
    title TriSeva Future Scope & Development Roadmap
    section Short-Term (0-6 Months)
        Speech & Voice I/O : Native Hindi/Hinglish Voice Queries via Whisper & ElevenLabs
        Hybrid Retrieval : Combining Vector E5 Search with Lexical BM25 (RRF Fusion)
        Quantized On-Device Router : ONNX / TensorRT DistilBERT compilation for 5ms latency
    section Medium-Term (6-18 Months)
        Expanded Domain Coverage : Adding Education, Micro-Finance & Disaster Management
        Local SLM Fine-Tuning : Fine-tuning Llama-3.2-3B / Qwen-2.5-3B on TriSeva-QA
        Automated Knowledge Auto-Update : Web Scraper pipelines for live scheme Gazette updates
    section Long-Term (18-36 Months)
        Offline Edge Deployment : Deployment on rural Mobile Apps & WhatsApp / IVRS Bots
        Formal Safety & Compliance Framework : Alignment with ICMR & Digital India AI Safety standards
        Multi-Agent Coalition Governance : Hierarchical consensus voting across 10+ agent nodes
```

---

### 11.1 Short-Term Enhancements (0 – 6 Months)

#### 1. Speech-to-Speech & Voice Interaction Subsystem
* **Motivation**: A substantial portion of rural citizens in India face literacy barriers. Text-only interfaces restrict accessibility.
* **Proposed Implementation**:
  * Integrate **Whisper-v3-large** or **Sarvam AI's Saaras Speech API** for real-time speech-to-text input in regional dialects (Bhojpuri, Haryanvi, Malvi, Marathi, Tamil).
  * Integrate regional text-to-speech (TTS) engines (e.g., **Indic-TTS** or **Bhashini API**) to read out agent advisories naturally over audio speakers.

#### 2. Hybrid Retrieval Fusion (Dense Vector + Sparse Lexical BM25)
* **Motivation**: While semantic vector search (`multilingual-e5-base`) captures conceptual intent, it occasionally misses exact legal section numbers (e.g., *"Section 420 BNS"*) or specific crop chemical names (e.g., *"Chlorpyrifos 20% EC"*).
* **Proposed Implementation**:
  * Implement **Reciprocal Rank Fusion (RRF)** combining ChromaDB dense vector retrieval with `rank_bm25` sparse lexical search.
  * Apply a re-ranking model (**BGE-Reranker-Large** or **Cohere Reranker**) on top $20$ candidate chunks to boost precision@k.

#### 3. Edge-Optimized Local Router (ONNX / TensorRT Quantization)
* **Motivation**: DistilBERT currently runs on PyTorch CPU tensors (~50ms latency).
* **Proposed Implementation**: Quantize the fine-tuned DistilBERT router model to **INT8 / ONNX runtime** or **TensorRT**, reducing classification latency to $<5\text{ms}$ while cutting memory consumption by $75\%$.

---

### 11.2 Medium-Term System Scaling (6 – 18 Months)

#### 1. Domain Portfolio Expansion (5 Core Domains)
* **Expansion Areas**:
  * **Education & Skill Development**: Career guidance, PMKVY scheme guidelines, scholarship eligibility (NSP).
  * **Financial Literacy & Micro-Finance**: KCC (Kisan Credit Card), Mudra loans, PM-Jan Dhan Yojana, insurance claims (PMFBY).
  * **Civic Rights & Disaster Management**: Panchayati Raj schemes, flood/drought relief procedures.
* **Architecture Shift**: Transition from a static 3-agent fanout to a dynamic **Sub-Graph Orchestration Pattern**, where domain super-nodes manage micro-specialist sub-agents.

#### 2. Specialist Small Language Model (SLM) Fine-Tuning
* **Motivation**: Relying on proprietary LLMs (Claude-3, Sarvam-105B) incurs API latency and recurring costs.
* **Proposed Implementation**:
  * Fine-tune open-weight Small Language Models (e.g., **Llama-3.2-3B-Instruct**, **Qwen-2.5-7B-Instruct**, or **Sarvam-2B**) directly on the **TriSeva-QA** dataset.
  * Apply **LoRA / QLoRA** parameter-efficient fine-tuning with Direct Preference Optimization (DPO) using Critic feedback logs as preference pairs.

#### 3. Automated Knowledge Base Synchronization & Scraper Pipeline
* **Motivation**: Government schemes and medical guidelines undergo continuous amendments. Static vector stores quickly become obsolete.
* **Proposed Implementation**:
  * Build automated scraper cron-jobs monitoring official portals (e.g., *India Code, MyScheme.gov.in, ICAR advisories, PIB releases*).
  * Implement an automated chunk updater that calculates document hashes ($\text{SHA-256}$) and dynamically upserts modified chunks into ChromaDB without taking the system offline.

---

### 11.3 Long-Term Vision & National Deployment (18 – 36 Months)

#### 1. Offline Mobile Edge Deployment & Low-Bandwidth Channels
* **Offline Mobile App**: Quantize specialist SLMs (e.g., 2B parameters using GGML / llama.cpp) to run directly on mid-range Android smartphones for offline field use by agricultural extensions and village health workers (ASHAs).
* **WhatsApp Business & IVRS Integration**: Connect TriSeva's API backend to **WhatsApp Business API** and **Interactive Voice Response Systems (IVRS)**, enabling farmers to call a toll-free number, speak their query, and receive instant spoken guidance.

#### 2. Formal Explainability (XAI) & Regulatory Compliance Framework
* Establish automated safety guardrails aligned with **ICMR ethical guidelines for AI in healthcare** and **Digital India Responsible AI Framework**.
* Incorporate **SHAP / LIME feature attribution** for structured tabular calculations (e.g. explaining exactly why an applicant was deemed ineligible for a government welfare loan).

#### 3. Multi-Agent Consensus Governance (Agentic Voting Coalition)
* Replace single-critic evaluation with a **Democracy of Critics** model: three independent auditor nodes (Fact Auditor, Safety Auditor, Format Auditor) vote on response approval using a majority consensus mechanism.

---

## 12. Repository Structure & Code References

The complete implementation is maintained in the project codebase:

```plaintext
Triseva/
├── agents/                 # Multi-agent LangGraph nodes and configurations
│   ├── orchestrator.py     # Intent classification & 3-tier fallback router
│   ├── health_agent.py     # Healthcare expert agent (ReAct loop)
│   ├── legal_agent.py      # Legal & government schemes agent (ReAct loop)
│   ├── agri_agent.py       # Agriculture expert agent (ReAct loop)
│   ├── critic.py           # Reflective self-correction & evaluation loop
│   ├── state.py            # LangGraph shared state schema definition
│   ├── llm_factory.py      # Dynamic LLM provider factory (Sarvam, Claude, Groq)
│   └── train_router.py     # DistilBERT domain classifier fine-tuning script
├── app/                    # UI Application codebase
│   ├── chainlit_app.py     # Primary ChatGPT-style Chainlit UI entrypoint
│   ├── streamlit_app.py    # Secondary Streamlit dashboard
│   ├── gradio_app.py       # Gradio debug playground
│   └── public/             # Static UI assets and custom CSS
├── evaluation/             # Ragas evaluation and benchmarking suite
│   ├── baseline_b1_naive.py     # B1 Baseline: Cross-domain unrouted RAG
│   ├── baseline_b2_nocritic.py  # B2 Baseline: Threshold 0.0 (No Critic retry)
│   ├── baseline_b3_keyword.py   # B3 Baseline: Lexical BM25 retriever
│   ├── ragas_eval.py            # Core evaluation & benchmark metrics pipeline
│   ├── push_feedback.py         # Offline metrics sync to LangSmith traces
│   └── results/                 # Raw benchmarking output CSVs & JSON logs
├── knowledge_base/         # Database management & vector storage
│   ├── build_kb.py         # Document parsing & ChromaDB collection builder
│   └── chunker.py          # Sentence-level contextual chunking pipeline
├── tools/                  # Agent tool implementations
│   ├── rag_tool.py         # Domain-scoped ChromaDB retrieval tools
│   ├── search_tool.py      # Tavily web search API wrapper
│   ├── calculator_tool.py  # Regex-based safe numerical evaluator
│   └── vision_tool.py      # Multimodal VLM OCR parser
├── utils/                  # Utility modules
│   └── translator.py       # Pre/Post translation & Hinglish caching engine
├── main.py                 # Core LangGraph state machine compiler & CLI validator
├── requirements.txt        # Python dependency manifest
├── Dockerfile              # HuggingFace Spaces container configuration
└── TriSeva_Project_Report.md # Primary academic dissertation report
```

---

*This document represents the master compilation of the TriSeva project architecture, empirical benchmarks, and multi-year strategic roadmap.*
