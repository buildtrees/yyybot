from __future__ import annotations

import asyncio
import json
import sys
import time
from types import SimpleNamespace

import pytest

from yyybot import ToolCall, ToolRegistry, web_fetch, web_search
from yyybot.tools import ToolError


def test_sync_and_async_tools_share_one_interface():
    registry = ToolRegistry()

    def sync_tool(value: int) -> int:
        """Double a value."""
        return value * 2

    async def async_tool(value: int) -> int:
        """Triple a value."""
        return value * 3

    sync = registry.add(sync_tool)
    registry.add(async_tool)

    assert sync.spec.parameters["properties"]["value"]["type"] == "integer"
    assert asyncio.run(registry.execute(ToolCall("1", "sync_tool", {"value": 2}))) == 4
    assert asyncio.run(registry.execute(ToolCall("2", "async_tool", {"value": 2}))) == 6


def test_slow_sync_tool_returns_after_worker_thread_finishes():
    registry = ToolRegistry()

    def slow_tool() -> str:
        """Return after enough time for the event loop to enter its wait cycle."""

        time.sleep(0.05)
        return "done"

    registry.add(slow_tool)

    assert asyncio.run(registry.execute(ToolCall("1", "slow_tool", {}))) == "done"


def test_web_search_uses_ddgs_and_formats_results(monkeypatch):
    calls = []

    class FakeDDGS:
        def __init__(self, **options):
            calls.append(options)

        def text(self, query, *, max_results, backend):
            calls.append(
                {
                    "query": query,
                    "max_results": max_results,
                    "backend": backend,
                }
            )
            return [
                {
                    "title": "<b>Example</b>",
                    "href": "https://example.com",
                    "body": "A &amp; B",
                }
            ]

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))
    monkeypatch.setenv("YYYBOT_WEB_PROXY", "http://proxy.example:8080")

    result = asyncio.run(web_search(" example ", count=99))

    assert calls == [
        {"timeout": 10, "proxy": "http://proxy.example:8080"},
        {"query": "example", "max_results": 10, "backend": "google"},
    ]
    assert result == (
        "Results for: example\n\n"
        "1. Example\n"
        "   https://example.com\n"
        "   A & B"
    )


def test_web_search_registers_with_expected_schema():
    registry = ToolRegistry()

    tool = registry.add(web_search)

    assert tool.spec.name == "web_search"
    assert tool.spec.parameters == {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["query"],
    }


def test_web_search_falls_back_when_google_fails(monkeypatch):
    calls = []

    class FakeDDGS:
        def __init__(self, **options):
            pass

        def text(self, query, *, max_results, backend):
            calls.append(backend)
            if backend == "google":
                raise RuntimeError("blocked")
            return [
                {
                    "title": "Fallback",
                    "href": "https://example.com/fallback",
                    "body": "Found elsewhere",
                }
            ]

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))

    result = asyncio.run(web_search("example"))

    assert calls == ["google", "duckduckgo"]
    assert "Fallback" in result


def test_web_fetch_extracts_html_and_uses_explicit_proxy(monkeypatch):
    client_options = []
    response_closed = []

    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}
        url = "https://example.com/article"
        encoding = "utf-8"

        def raise_for_status(self):
            return None

        async def aiter_bytes(self):
            yield (
                b"<html><head><title>Example</title><script>ignore()</script></head>"
                b"<body><h1>Heading</h1><p>Hello &amp; goodbye.</p></body></html>"
            )

        async def aclose(self):
            response_closed.append(True)

    class FakeClient:
        def __init__(self, **options):
            client_options.append(options)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def build_request(self, method, url, *, headers):
            return {"method": method, "url": url, "headers": headers}

        async def send(self, request, *, stream):
            assert request["url"] == "https://example.com/article"
            assert stream is True
            return FakeResponse()

    import httpx

    monkeypatch.setattr("yyybot.web._validate_public_url", lambda url: None)
    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    monkeypatch.setenv("YYYBOT_WEB_PROXY", "http://proxy.example:8080")

    result = json.loads(
        asyncio.run(web_fetch("https://example.com/article", max_chars=1_000))
    )

    assert client_options == [
        {
            "proxy": "http://proxy.example:8080",
            "timeout": 30.0,
            "follow_redirects": False,
            "trust_env": False,
        }
    ]
    assert result["extractor"] == "html"
    assert result["untrusted"] is True
    assert "# Example" in result["text"]
    assert "Heading" in result["text"]
    assert "Hello & goodbye." in result["text"]
    assert "ignore()" not in result["text"]
    assert response_closed == [True]


def test_web_fetch_rejects_private_network_targets():
    with pytest.raises(ToolError, match="non-public"):
        asyncio.run(web_fetch("http://127.0.0.1/private"))


def test_web_fetch_registers_with_expected_schema():
    registry = ToolRegistry()

    tool = registry.add(web_fetch)

    assert tool.spec.name == "web_fetch"
    assert tool.spec.parameters == {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "max_chars": {"type": "integer"},
        },
        "required": ["url"],
    }
