"""The app (FastAPI): its security and sessions, the route modules (``routes``) and the built React frontend
(``static/``).

- **Hosts:** only ``settings.allowed_hosts`` (loopback names in real mode; DNS rebinding is refused).
- **Writes:** cross-origin writes are refused (``apps.common.security.cross_origin_write``).
- **Sign-in:** with an access token (always in real mode), every path except ``/login`` and the frontend's hashed
  assets needs the sign-in cookie or a bearer token.
- **Sessions:** every response sets a random session cookie; jobs, model connections, the patient workspace and
  populations belong to it (``routes.common.owner``).
- **Frames:** no other site may frame the app, except the origins of ``settings.frame_ancestors`` (synthetic mode,
  e.g. a Hugging Face Space page); over https (here or at a proxy) the session cookie is then partitioned
  (SameSite=None, Secure, Partitioned), the form browsers send to a framed site.
- **Errors:** bodies are ``{"error": message}``; ``InputError`` messages are written for the user (400),
  ``BusyError`` means a job is already running for the session (409).
"""
from __future__ import annotations

from pathlib import Path
import re
import secrets

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException as FastAPIHTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from matchminer_ai.apps.common import security

from .core.encoder import EncoderUnavailable
from .core.errors import BusyError, ConflictError, InputError
from .evidence import CatalogUnavailable, UnknownEntry
from .routes import (assistant, benchmarking, docs, guidelines, local_server, management, patients, population, records,
                     status, trials)
from .state import DashState, NotFound

STATIC = Path(__file__).with_name("static")
CSP = security.content_security_policy(inline_styles=True, form_action="'self'")  # the sign-in form posts here
HEADERS = security.security_headers(CSP)
AUTH_COOKIE = "mmai_dash_auth"
SESSION_COOKIE = "mmai_dash_session"
SESSION_LENGTH = 43  # secrets.token_urlsafe(32)
SESSION_TOKEN = re.compile(rf"[A-Za-z0-9_-]{{{SESSION_LENGTH}}}")
LOGIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Sign in · MatchMiner-AI dashboard</title></head>
<body><main><h1>MatchMiner-AI dashboard</h1>{error}
<form method="post" action="/login"><label for="token">Access token</label>
<input id="token" name="token" type="password" autocomplete="current-password" required>
<button type="submit">Sign in</button></form>
<p>The token is printed by <code>mmai-dashboard</code> when it starts in a terminal, or kept in its access token
file.</p>
</main></body></html>
"""


def create_app(state: DashState, access_token: str | None = None) -> FastAPI:
    """The app: allowed Host names only, no cross-origin writes, browser sessions and, with ``access_token``,
    sign-in."""
    from fastapi.responses import HTMLResponse, RedirectResponse
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    app = FastAPI(title="MatchMiner-AI dashboard (research use)", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.mmai = state
    framed = tuple(getattr(state.settings, "frame_ancestors", ()) or ())
    headers_for_all = (security.security_headers(security.content_security_policy(
        inline_styles=True, form_action="'self'", frame_ancestors=framed)) if framed else HEADERS)
    sign_in = security.SignIn(access_token, b"mmai-dashboard browser sign-in v1", AUTH_COOKIE) if access_token else None

    @app.middleware("http")
    async def headers(request: Request, call_next):
        refused = security.cross_origin_write(request.method, request.headers)
        path = request.url.path
        if refused:
            response = JSONResponse({"error": refused}, status_code=403)
        elif sign_in and path != "/login" and not path.startswith("/assets/") and not sign_in.authorized(request):
            response = (JSONResponse({"error": "Sign in with the access token."}, status_code=401)
                        if path.startswith("/api/") else RedirectResponse("/login", status_code=303))
        else:
            session = request.cookies.get(SESSION_COOKIE)
            if not session or not SESSION_TOKEN.fullmatch(session):
                session = secrets.token_urlsafe(32)
            request.state.owner = session
            response = await call_next(request)
            # https here or at the proxy in front (a Space's proxy says so in X-Forwarded-Proto).
            https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
            if framed and https:
                # In another site's frame this is a third-party cookie: browsers keep it only with SameSite=None,
                # Secure and, increasingly, Partitioned. Written by hand: Python's cookie module knows Partitioned
                # only from 3.14. Plain http (a local test) keeps the strict cookie, which an http client sends back.
                response.headers.append("set-cookie", f"{SESSION_COOKIE}={session}; HttpOnly; Path=/; SameSite=none; "
                                                      "Secure; Partitioned")
            else:
                response.set_cookie(SESSION_COOKIE, session, httponly=True, samesite="strict", secure=https)
        response.headers.update(headers_for_all)
        if path.startswith("/api/") or path == "/login":
            response.headers["Cache-Control"] = "no-store"
        elif path.startswith("/assets/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"  # hashed file names
        return response

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(state.settings.allowed_hosts))

    if sign_in:
        def login_page(failed: bool = False):
            error = "<p role='alert'>That access token was not accepted.</p>" if failed else ""
            return HTMLResponse(LOGIN_PAGE.replace("{error}", error), status_code=401 if failed else 200)

        def signed_in(request: Request, token: str):
            if not sign_in.valid_token(token):
                return login_page(failed=True)
            return sign_in.set_cookie(request, RedirectResponse("/", status_code=303))

        @app.get("/login")
        def login_form(request: Request, token: str | None = None):
            # mmai-dashboard prints /login?token=... to its own terminal for one-click sign-in.
            return signed_in(request, token) if token is not None else login_page()

        @app.post("/login")
        async def login(request: Request):
            from urllib.parse import parse_qs

            body = (await request.body())[:2000].decode("utf-8", "replace")
            return signed_in(request, (parse_qs(body).get("token") or [""])[0])

        @app.post("/api/logout")
        def logout():
            response = JSONResponse({"signed_out": True})
            response.delete_cookie(AUTH_COOKIE, httponly=True, samesite="strict")
            return response

    def error(status_code: int, message: str) -> JSONResponse:
        return JSONResponse({"error": message}, status_code=status_code)

    @app.exception_handler(NotFound)
    async def not_found(_, exc):
        return error(404, f"unknown id {exc.args[0]!r}")

    @app.exception_handler(UnknownEntry)
    async def unknown_entry(_, exc):
        return error(404, f"the GoodOption catalog has no {exc.args[0]!r}")

    @app.exception_handler(CatalogUnavailable)
    async def no_catalog(_, exc):
        return error(404, str(exc))

    @app.exception_handler(EncoderUnavailable)
    async def encoder_off(_, exc):
        return error(409, f"the TrialSpace encoder is unavailable: {exc}")

    @app.exception_handler(InputError)
    async def bad_input(_, exc):
        return error(400, str(exc))

    @app.exception_handler(BusyError)
    async def busy(_, exc):
        return error(409, str(exc))

    @app.exception_handler(ConflictError)
    async def conflict(_, exc):
        return error(409, str(exc))

    @app.exception_handler(ValueError)
    async def bad_value(_, exc):
        return error(400, str(exc))

    @app.exception_handler(FastAPIHTTPException)
    async def http_error(_, exc):
        return error(exc.status_code, str(exc.detail))

    for module in (status, trials, management, patients, records, assistant):
        app.include_router(module.router(state))
    for module in (guidelines, population):
        app.include_router(module.router(state.guidelines))
    app.include_router(benchmarking.router(state.benchmarks))
    app.include_router(docs.router())
    app.include_router(local_server.router(state.local_server))

    # ------------------------------------------------------------------ the frontend (after every API route)

    if (STATIC / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            return error(404, "not found")
        file = (STATIC / path).resolve()
        if path and file.is_file() and STATIC.resolve() in file.parents:
            return FileResponse(file)
        index = STATIC / "index.html"
        if not index.exists():
            return error(503, "frontend not built: cd frontend/dashboard && pnpm build")
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return app
