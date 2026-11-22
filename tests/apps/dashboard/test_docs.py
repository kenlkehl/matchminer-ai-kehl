"""The How this works tab: only the curated documentation pages are served, and the extraction data model page is
generated from the schema declarations."""
from __future__ import annotations


def test_the_docs_route_serves_the_curated_pages_and_nothing_else():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from matchminer_ai.apps.dashboard import docs
    from matchminer_ai.apps.dashboard.routes import docs as route

    app = FastAPI()
    app.include_router(route.router())
    client = TestClient(app)
    listed = [p["id"] for p in client.get("/api/docs").json()["pages"]]
    assert listed == [p["id"] for p in docs.pages()] and "matching" in listed
    for page_id in listed:
        body = client.get(f"/api/docs/{page_id}").json()
        assert body["markdown"].startswith("# ") and ":::" not in body["markdown"]
        assert "```mermaid" not in body["markdown"]
    matching = client.get("/api/docs/matching").json()["markdown"]
    assert "(page:training)" in matching and "../training/index.md" not in matching  # in-tab links
    for other in ("index", "..%2F..%2Fmkdocs.yml", "matching.md", "%2Fetc%2Fpasswd"):
        assert client.get(f"/api/docs/{other}").status_code == 404


def test_the_data_model_page_lists_every_schema_item_type():
    from matchminer_ai.apps.dashboard.docs import page
    from matchminer_ai.patients.extraction.extensions import registered_extensions
    from matchminer_ai.patients.extraction.schema import CORE_ITEM_TYPES, SCHEMA_VERSION

    markdown = page("extraction-data-model")["markdown"]
    assert SCHEMA_VERSION in markdown
    item_types = [*CORE_ITEM_TYPES, *(t for e in registered_extensions().values() for t in e.item_types)]
    for item_type in item_types:
        assert f"### `{item_type.key}`" in markdown
        for field in item_type.fields:
            assert f"| `{field.name}` |" in markdown
