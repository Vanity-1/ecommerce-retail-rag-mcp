# ChromaDB Schema and Data Ingestion Architecture (Mermaid, with ADK VectorSearchTool)

```mermaid
flowchart TB
  %% Sources
  subgraph Sources[Data Sources]
    S1[Product Catalog]
    S2[Reviews]
    S3[Policies / FAQs]
    S4[Support Tickets]
    S5[Inventory / Orders]
  end

  %% Ingestion
  subgraph Ingest[Preprocess & Chunk]
    C1[Cleaning / Normalize]
    C2[Field-Aware Chunking]
    C3[Lang Tags / Metadata]
  end

  %% Embedding
  subgraph Embed[Embedding]
    E1[embed_text<br/>(Ollama Gemma or sentence-embedder)<br/>Vectorize Chunks]
  end

  %% Chroma collections
  subgraph Chroma[ChromaDB Collections]
    CC1[catalog_chunks<br/>id, embedding(768), sku, field, lang, brand, category, price, stock, text]
    CC2[policy_chunks<br/>id, embedding, topic, lang, text]
    CC3[review_chunks<br/>id, embedding, sku, rating, lang, text]
    CC4[ticket_chunks<br/>id, embedding, customer_id, lang, text]
  end

  %% Query path (ADK tool)
  subgraph Query[ADK VectorSearchTool]
    Q1[Query Embedding]
    Q2[KNN + Filters]
    Q3[Rerank / Score Merge]
  end

  S1 --> C1
  S2 --> C1
  S3 --> C1
  S4 --> C1
  S5 --> C1

  C1 --> C2
  C2 --> C3
  C3 --> E1
  E1 --> CC1
  E1 --> CC2
  E1 --> CC3
  E1 --> CC4

  Q1 --> Q2
  Q2 --> Q3
  Q2 --> CC1
  Q2 --> CC2
  Q2 --> CC3
  Q2 --> CC4
```
