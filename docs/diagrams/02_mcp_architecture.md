# MCP Architecture Diagram: Retail RAG System Tool Integration (Mermaid, with ADK)

```mermaid
flowchart TB
  %% Host / Client Layer
  subgraph Host[Host / Client Layer]
    GW[API Gateway / App]
    AUTH[AuthN/Z & Consent]
    SESS[Session & Context Store]
  end

  %% ADK Orchestration
  subgraph ADK[Google ADK]
    AG[RetailRAGAgent / Workflows]
    ATS[ADK Tools: VectorSearch, Generate, MCP Wrappers]
  end

  %% MCP Protocol
  subgraph Protocol[MCP Protocol]
    INIT[Initialize]
    CALL[Tool Call]
    RESP[Tool Response]
    ERR[Error / Retry]
  end

  %% MCP Servers
  subgraph Servers[MCP Servers]
    REG[Tool Registry & Routing]
    SRV1[product_search]
    SRV2[inventory_check]
    SRV3[policy_qa]
    SRV4[analytics_query]
    LOG[Observability: Logs / Metrics / Tracing]
  end

  %% Data Systems
  subgraph Data[External / Data Systems]
    CAT[ChromaDB (Vectors)]
    INV[Inventory / Orders]
    POL[Policies / FAQs]
    DWH[Analytics Warehouse]
    OLM[Ollama (Gemma3:270m)]
  end

  GW --> AUTH
  AUTH --> SESS
  SESS --> AG
  AG --> ATS
  ATS --> OLM
  ATS --> CAT
  ATS --> REG

  GW --> INIT
  INIT --> REG

  REG --> CALL
  CALL --> SRV1
  CALL --> SRV2
  CALL --> SRV3
  CALL --> SRV4

  SRV1 --> CAT
  SRV2 --> INV
  SRV3 --> POL
  SRV4 --> DWH

  SRV1 --> RESP
  SRV2 --> RESP
  SRV3 --> RESP
  SRV4 --> RESP
  RESP --> GW

  SRV1 --> LOG
  SRV2 --> LOG
  SRV3 --> LOG
  SRV4 --> LOG

  CALL --> ERR
  ERR --> CALL
```
