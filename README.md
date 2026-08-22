# E-commerce Retail RAG + MCP 客服系统

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![RAG](https://img.shields.io/badge/RAG-多源融合-red.svg)]() [![MCP](https://img.shields.io/badge/Protocol-MCP-green.svg)]() [![ADK](https://img.shields.io/badge/Google-ADK-orange.svg)]() [![LLM](https://img.shields.io/badge/LLM-Ollama-8A2BE2.svg)]()

一个用于电商/零售场景的**多源检索增强生成（RAG）客服系统**。基于 Google ADK 编排 Agent，检索商品目录与退货/配送政策，并经 **RRF（Reciprocal Rank Fusion）多源融合**排序后，让 LLM 基于**带出处的上下文**生成可靠回答，杜绝幻觉。

## ✨ 核心能力

| 能力 | 说明 | 状态 |
|------|------|------|
| 多源 RAG 检索 | 商品目录 + 售后政策双源检索，单次嵌入、多源召回 | ✅ 可用 |
| RRF 融合排序 | 解决数据量不均衡下「强源淹没弱源」问题 | ✅ 已实验验证 |
| 端到端 Demo | Web 界面：用户问 → 多源检索 → 带引用回答 | ✅ 现场可跑 |
| MCP 工具封装 | product_search / policy_qa 等工具经 MCP 暴露 | ✅ 已实现 |
| 嵌入一致性守卫 | 统一 `nomic-embed-text`(768维)，测试兜底 | ✅ 已内置 |

## 📊 二次开发成果（可直接写进简历）

**RRF 融合对比实验** —— 相同 12 条查询，对比「RRF 融合」与「直接拼接」的召回效果：

| 指标 | 直接拼接 | RRF 融合 |
|------|:---:|:---:|
| 政策相关查询 Top-3 相关命中率 | **9/12** | **12/12** |
| 弱源(政策)高相关文档排名 | 被商品目录淹没 | 提升至第 2 位 |

> 结论：在商品目录(强源)体量远大于政策(弱源)时，直接拼接会让政策文档完全被淹没；RRF 融合后相关文档显著上浮。完整分析见 [`docs/RRF对比实验报告.html`](docs/RRF对比实验报告.html)。

**踩坑记录**（面试加分点）→ [`docs/RAG二开踩坑记录.html`](docs/RAG二开踩坑记录.html)

## 🚀 快速开始 —— 现场跑 Demo

仅需本地 [Ollama](https://ollama.com/)，无需云 API Key。**5 分钟可跑通**。

```bash
# 1. 启动 Ollama 并拉取所需模型（首次）
ollama serve &
ollama pull nomic-embed-text      # 嵌入模型（768维）
ollama pull qwen2.5:3b            # 生成模型

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3.（可选）如预计算嵌入缓存缺失则重建向量库
python scripts/_precompute_embeddings.py

# 4. 启动端到端 Web Demo
python scripts/demo_web.py
# 浏览器打开 http://127.0.0.1:8080
```

试问：`do you refund damaged produce`、`free delivery over 35 dollars`。

> Demo 的检索依赖预计算嵌入缓存 `embedding_index.json + embeddings_cache.npy`（由 Ollama 生成），已内置内存余弦检索层，无需额外启动 ChromaDB，避免 Windows 下单机持久化的坑。

## 🧪 复现对比实验

```bash
python scripts/compare_rrf_vs_join.py
```

## 💻 技术栈

- **编排**：Google ADK（`FunctionTool` + `SequentialAgent`）
- **嵌入/生成**：Ollama（`nomic-embed-text` / `qwen2.5:3b`），纯本地
- **向量检索**：预计算嵌入缓存 + NumPy 内存余弦检索（原 ChromaDB 适配层保留）
- **融合排序**：自研 RRF（`agents/tools/rrf_fusion.py`）
- **服务**：FastAPI MCP server（`src/mcp_server`）

## 📁 目录结构

```
scripts/
  demo_web.py                  # 端到端 Web Demo（问→检→带出处答）
  compare_rrf_vs_join.py       # RRF vs 直接拼接对比实验
  _precompute_embeddings.py    # 预计算嵌入缓存
  chat_rag_demo.py             # 命令行 RAG Demo
src/
  embeddings/ollama_client.py  # embed_text + generate_answer
  rag/
    inmemory_cache.py          # 内存余弦检索层（二开新增）
    vector_store.py            # ChromaDB 适配层（保留）
agents/
  tools/rrf_fusion.py          # RRF 融合算法（二开新增）
  rag_agent.py                 # ADK RAG Agent
  workflows.py                 # ADK 工作流
docs/
  RRF对比实验报告.html          # 二开：对比实验报告
  RAG二开踩坑记录.html          # 二开：踩坑技术博客
configs/                       # 环境变量模板
tests/                         # 含嵌入一致性守卫测试
```

## ✅ 测试

```bash
pytest tests/            # 单元测试 + 嵌入一致性守卫
bash scripts/checks.sh   # 静态检查
```

## 🌱 后续方向

- 接入更多检索源（评论、库存）与重排序 Reranker
- 增加评估管线（recall@k / groundedness）
- MCP 工具的 RBAC / 限流