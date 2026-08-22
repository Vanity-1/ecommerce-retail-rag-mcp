import os
import httpx

API_BASE = os.getenv("API_BASE", "http://localhost:8000")


async def call_mcp_tool(name: str, arguments: dict) -> dict:
    """Invoke a tool exposed by the FastAPI MCP server."""
    payload = {
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    async with httpx.AsyncClient(timeout=30) as ac:
        r = await ac.post(f"{API_BASE}/mcp/tools/call", json=payload)
        r.raise_for_status()
        return r.json()


async def product_search(query: str, top_k: int = 5) -> dict:
    """ADK FunctionTool entrypoint: search products exposed by the MCP server."""
    return await call_mcp_tool("product_search", {"query": query, "top_k": top_k})


async def inventory_check(sku: str) -> dict:
    """ADK FunctionTool entrypoint: check inventory for a SKU."""
    return await call_mcp_tool("inventory_check", {"sku": sku})


async def policy_qa(query: str) -> dict:
    """ADK FunctionTool entrypoint: answer a policy question."""
    return await call_mcp_tool("policy_qa", {"query": query})


async def analytics_query(range_: str = "7d") -> dict:
    """ADK FunctionTool entrypoint: query analytics."""
    return await call_mcp_tool("analytics_query", {"range": range_})


# ---------------------------------------------------------------------------
# Compatibility wrappers (old-style Tool classes) for any code still relying on
# the previous API. New ADK code should import the plain functions above and
# wrap them with google.adk FunctionTool.
# ---------------------------------------------------------------------------
class ProductSearchTool:
    async def call(self, ctx):
        return await product_search(ctx.state.get("query", ""))


class InventoryCheckTool:
    async def call(self, ctx):
        return await inventory_check(ctx.state.get("sku", ""))


class PolicyQATool:
    async def call(self, ctx):
        return await policy_qa(ctx.state.get("query", ""))


class AnalyticsQueryTool:
    async def call(self, ctx):
        return await analytics_query(ctx.state.get("range", "7d"))