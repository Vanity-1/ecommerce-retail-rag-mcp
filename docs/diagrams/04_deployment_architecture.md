# Deployment Architecture: Local to Production Environments (Mermaid, with ADK Runner)

```mermaid
flowchart TB
  %% Dev
  subgraph Dev[Local Dev (Docker Compose)]
    D1[Ollama: gemma3:270m]
    D2[ChromaDB]
    D3[MCP Server (FastAPI)]
    D4[API Gateway]
    D5[ADK Runner (Agents)]
  end

  %% Staging
  subgraph Staging[Staging]
    S1[Ollama]
    S2[ChromaDB]
    S3[MCP Server]
    S4[API Gateway]
    S5[CI / CD Deploy]
    S6[Prometheus / Grafana / Logs]
    S7[ADK Runner]
  end

  %% Production
  subgraph Prod[Production (K8s / Cloud)]
    P1[LB / Ingress]
    P2[Ollama Auto-Scale]
    P3[ChromaDB HA]
    P4[MCP Server ASG]
    P5[API Gateway / WAF]
    P6[Observability Stack]
    P7[Secrets / Key Mgmt]
    P8[Backup / DR]
    P9[ADK Runner ASG]
  end

  Dev --> Staging
  Staging --> Prod

  P1 --> P5
  P5 --> P4
  P4 --> P2
  P4 --> P3
  P4 --> P9

  P4 --> P6
  P2 --> P6
  P3 --> P6
  P9 --> P6

  P4 --> P7
  P2 --> P7
  P3 --> P7
  P9 --> P7

  P3 --> P8
```
