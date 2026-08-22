#!/usr/bin/env python3
"""
End-to-end RAG demo web UI (self-contained, stdlib only).

Shows the full "user ask -> multi-source retrieve -> grounded answer" loop with
source citations. Uses:
  - Ollama nomic-embed-text for query embedding
  - Chroma local persistent store (catalog_chunks + policies)
  - RRF fusion to rank combined results across sources
  - Ollama generation model (qwen2.5:3b) for the grounded answer

Run:
    python scripts/demo_web.py

Then open http://127.0.0.1:8080 in a browser.
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("OLLAMA_EMBED_MODEL", "nomic-embed-text")

from src.embeddings.ollama_client import embed_text, generate_answer
from src.rag import inmemory_cache
from agents.tools.rrf_fusion import fuse

COLLECTIONS = ("catalog_chunks", "policies")
GEN_MODEL = os.getenv("OLLAMA_GEN_MODEL", "qwen2.5:3b")

_store_lock = threading.Lock()


class Simulator:
    """Holds retrieval + generation so the HTTP handler stays thin."""

    def __init__(self):
        inmemory_cache.load()  # eager-load the cache once

    # ----- retrieval (single embed, then per-source top-k via in-memory cache) -----
    def _retrieve_groups(self, query, k=4):
        qv = embed_text(query)
        groups = []
        for name in COLLECTIONS:
            res = inmemory_cache.cosine_search(name, qv, k=k)
            groups.append([
                {"collection": name, "text": res["documents"][i],
                 "meta": res["metadatas"][i], "id": res["ids"][i]}
                for i in range(len(res["ids"]))
            ])
        return groups

    # ----- the two fusion modes we compare in the UI -----
    def retrieve_rrf(self, query, n_final=6):
        groups = self._retrieve_groups(query)
        fused = fuse(groups)[:n_final]
        # attach source label
        src_map = {}
        for g in groups:
            for d in g:
                src_map.setdefault(d["text"].strip().lower(), d)
        out = []
        for d in fused:
            key = d["text"].strip().lower()
            out.append({**d, **src_map.get(key, {})})
        return out

    def retrieve_plain(self, query, n_final=6):
        groups = self._retrieve_groups(query)
        seen, out = set(), []
        for g in groups:
            for d in g:
                key = d["text"].strip().lower()
                if key not in seen:
                    seen.add(key)
                    out.append(d)
        return out[:n_final]

    # ----- grounded answer with citations -----
    def answer(self, query, mode="rrf"):
        t0 = time.time()
        if mode == "rrf":
            docs = self.retrieve_rrf(query)
        else:
            docs = self.retrieve_plain(query)

        context_blocks = []
        for i, d in enumerate(docs, 1):
            meta = d.get("meta", {})
            src = d.get("collection", "?")
            label = meta.get("product_name")
            if not label:
                label = meta.get("type", src)
            context_blocks.append(f"[{i}] ({label})\n{d['text'].strip()}")
        context = "\n\n".join(context_blocks)

        prompt = (
            "You are a retail assistant. Answer the customer question using ONLY "
            "the provided product/policy context. Cite sources with [1], [2] ... "
            "matching the brackets in CONTEXT. If the context is not enough, say so "
            "and ask one short follow-up. Be concise.\n\n"
            f"QUESTION: {query}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"ANSWER:"
        )
        answer = generate_answer(prompt, model=GEN_MODEL)
        elapsed = time.time() - t0
        return {"query": query, "mode": mode, "answer": answer,
                "sources": docs, "elapsed_s": round(elapsed, 2)}


# ---------------------------------------------------------------------------
# Minimal HTTP layer (stdlib only)
# ---------------------------------------------------------------------------
PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>零售 RAG 客服 Demo</title>
<style>
  body{font-family:-apple-system,'Segoe UI',Roboto,'Microsoft YaHei',sans-serif;background:#0f172a;color:#e2e8f0;max-width:900px;margin:30px auto;padding:0 20px}
  h1{font-size:22px} .dim{color:#94a3b8;font-size:13px;margin-top:6px}
  .box{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:18px;margin-top:16px}
  input[type=text]{width:100%;padding:12px 14px;border-radius:8px;border:1px solid #475569;background:#0f172a;color:#e2e8f0;font-size:15px;box-sizing:border-box}
  select{padding:9px;border-radius:8px;border:1px solid #475569;background:#0f172a;color:#e2e8f0;margin-top:10px}
  button{margin-top:10px;padding:10px 20px;border-radius:8px;border:0;background:#38bdf8;color:#0f172a;font-weight:600;cursor:pointer}
  button:disabled{opacity:.5}
  .ans{white-space:pre-wrap;line-height:1.7;margin-top:6px}
  .meta{color:#94a3b8;font-size:12px;margin-top:10px}
  .badge{display:inline-block;font-size:11px;border-radius:5px;padding:2px 8px;margin-left:6px}
  .rrf{background:#065f46;color:#a7f3d0}.plain{background:#1e3a8a;color:#bfdbfe}
  .src{background:#0f172a;border-left:3px solid #38bdf8;padding:8px 12px;margin-top:8px;border-radius:0 6px 6px 0;font-size:13px;color:#cbd5e1}
  .src .tag{color:#38bdf8;font-weight:600}
</style></head><body>
  <h1>零售 RAG 客服系统</h1>
  <div class="dim">用户提问 → 多知识源检索（商品目录 + 退货政策）→ RRF 融合排序 → 生成带出处回答</div>
  <div class="box">
    <input id="q" type="text" placeholder="例如：有机牛油果多少钱？ / 生鲜坏了能退吗？ / 有冷冻披萨吗？">
    <div>
      <select id="mode">
        <option value="rrf">RRF 多源融合</option>
        <option value="plain">直接拼接（对照）</option>
      </select>
    </div>
    <button id="go" onclick="ask()">发送</button>
  </div>
  <div class="box" id="out" style="display:none">
    <div id="answerBox"></div>
    <div class="meta" id="metaBox"></div>
    <div id="srcBox"></div>
  </div>
<script>
async function ask(){
  var q=document.getElementById('q').value.trim();
  if(!q) return;
  var mode=document.getElementById('mode').value;
  var go=document.getElementById('go'); go.disabled=true;
  var out=document.getElementById('out'); out.style.display='block';
  document.getElementById('answerBox').innerHTML='<div class="meta">正在检索并生成…（首次检索约需数秒）</div>';
  document.getElementById('srcBox').innerHTML='';
  try{
    var r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({q:q,mode:mode})});
    var d=await r.json();
    var badge=mode==='rrf'?'<span class="badge rrf">RRF 融合</span>':'<span class="badge plain">直接拼接</span>';
    document.getElementById('answerBox').innerHTML='<div class="ans">'+d.answer+'</div>';
    document.getElementById('metaBox').innerHTML='问题：'+d.query+'　耗时 '+d.elapsed_s+' 秒　引用 '+d.sources.length+' 条　'+badge;
    var s='';
    d.sources.forEach(function(src,i){
      var label=(src.meta&&(src.meta.product_name||src.meta.type))||src.collection||'source';
      var txt=(src.text||'').substring(0,160);
      s+='<div class="src"><span class="tag">['+(i+1)+'] '+label+'</span><br>'+txt+'…</div>';
    });
    document.getElementById('srcBox').innerHTML=s;
  }catch(e){
    document.getElementById('answerBox').innerHTML='<div style="color:#f87171">出错：'+e+'</div>';
  }finally{go.disabled=false;}
}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    sim = None  # set after construction

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path != "/ask":
            self._send(404, "text/plain", b"not found")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            q = (body.get("q") or "").strip()
            mode = body.get("mode") or "rrf"
            if not q:
                raise ValueError("empty query")
            with _store_lock:
                result = self.sim.answer(q, mode=mode)
            self._send(200, "application/json; charset=utf-8", json.dumps(result, ensure_ascii=False).encode("utf-8"))
        except Exception as e:
            self._send(500, "application/json", json.dumps({"error": str(e)}).encode("utf-8"))

    def _send(self, code, ctype, data):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    sim = Simulator()
    Handler.sim = sim
    port = int(os.getenv("DEMO_PORT", "8080"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"RAG demo running at http://127.0.0.1:{port}  (store=in-memory cache)")
    server.serve_forever()


if __name__ == "__main__":
    main()