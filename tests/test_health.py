import os
from httpx import AsyncClient
import pytest

@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(base_url="http://localhost:8000") as ac:
        r = await ac.get("/healthz")
        assert r.status_code == 200
        assert r.json().get("status") == "ok"
