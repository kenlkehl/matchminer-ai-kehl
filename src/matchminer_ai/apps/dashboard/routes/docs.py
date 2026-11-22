"""The How this works tab: the curated documentation pages (``docs``); only listed page ids are served."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import docs


def router() -> APIRouter:
    api = APIRouter()

    @api.get("/api/docs")
    def index():
        return {"pages": docs.index()}

    @api.get("/api/docs/{page_id}")
    def page(page_id: str):
        try:
            return docs.page(page_id)
        except docs.UnknownPage:
            raise HTTPException(404, "There is no such documentation page.") from None

    return api
