# Data Flow Diagram: E-commerce RAG Pipeline (Mermaid, with ADK)

```mermaid
flowchart LR
  %% Client
  subgraph Client
    U[User Query]
  end

  %% ADK Orchestration
  subgraph ADK[Google ADK Orchestration]
    AG[RetailRAGAgent - Sequential]
    VS[VectorSearchTool]
    GT[GenerateTool]
  end

  %% API / MCP-facing preprocessing (optional)
  subgraph API[FastAPI MCP Server]
    N[Normalize / Lang Detect]
    JR[Join Runtime Context]
  end

  %% Embedding
  subgraph Embedding
    E[embed_text<br/>(Ollama Gemma3 embeddings<br/>or sentence-embedder)]
  end

  %% Vector DB
  subgraph VectorDB[ChromaDB]
    Q[Vector Search + Filters]
    R[Rerank / Score Merge]
  end

  %% Augmentation
  subgraph Augment
    C[Context Builder (top-k)]
  end

  %% Generation
  subgraph Gen[Generation]
    G[Ollama Gemma3:270m - Grounded Answer]
  end

  %% MCP Tool Layer
  subgraph MCP[MCP Tool Layer]
    T1[product_search]
    T2[inventory_check]
    T3[policy_qa]
    T4[analytics_query]
  end

  %% Main happy path
  U --> AG
  AG --> N
  N --> E
  E --> Q
  Q --> R
  R --> C
  C --> G
  G -->|Response| U

  %% ADK tool calls (logical)
  AG -->|uses| VS
  AG -->|uses| GT

  %% Context sufficiency branch
  C --> D{Enough Context?}
  D -- Yes --> G
  D -- No --> T1
  D -- No --> T2
  D -- No --> T3
  D -- No --> T4
  T1 --> JR
  T2 --> JR
  T3 --> JR
  T4 --> JR
  JR --> C

  %% Styling
  classDef stage fill:#eef,stroke:#88f,stroke-width:1px;
  class ADK,API,Embedding,VectorDB,Augment,Gen,MCP stage;
```
