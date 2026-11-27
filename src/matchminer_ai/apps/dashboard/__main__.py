"""``mmai-dashboard --mode real|synthetic`` (``python -m matchminer_ai.apps.dashboard``): serve the app;
``mmai-dashboard --mode ... build``: build its trial store once; ``mmai-dashboard --mode ... prepare``: only build the
synthetic run from its published dataset (``hub_run``; ``serve`` and ``build`` do it first too).

Real mode serves on loopback only and always asks for the access token; synthetic mode may serve beyond loopback
and asks for a token only with ``--require-token`` (``settings``)."""
import os
import signal
import sys
import threading

from matchminer_ai.apps.common.security import is_loopback

from .settings import parse_settings


def main(argv=None):
    from matchminer_ai.llm.openrouter import take_hosted_key

    take_hosted_key()  # the hosted OpenRouter key leaves the environment before anything else starts
    settings = parse_settings(argv)
    if settings.command == "serve" and settings.real and not is_loopback(settings.host):
        sys.exit(f"Refusing to serve on {settings.host}: real mode shows patient records and confidential protocol "
                 "content and serves on 127.0.0.1 only (forward the port with ssh -L to reach it from another machine).")
    if not settings.allow_downloads:  # before transformers or huggingface_hub is imported
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # A published synthetic run (the site config's synthetic_run names hf://datasets/...): build its folder first.
    from .hub_run import prepare

    prepare(settings, log=lambda message: print(message, flush=True))
    if settings.command == "prepare":
        return
    if settings.command == "build":
        from .pipeline import write_store
        from .state import make_builder

        store = make_builder(settings).build()
        write_store(store, settings.store_path)
        print(f"{store['counts']} -> {settings.store_path} ({store['build_seconds']} s)")
        return
    import uvicorn

    from .server import create_app
    from .state import DashState

    token = None
    if settings.require_token:
        from matchminer_ai.apps.common.security import load_access_token

        from .settings import ACCESS_TOKEN_ENV, ACCESS_TOKEN_FILE

        token = load_access_token(ACCESS_TOKEN_ENV, ACCESS_TOKEN_FILE)
        url = f"http://{settings.host}:{settings.port}/login"
        if sys.stdout.isatty():
            print(f"Sign in: {url}?token={token}", flush=True)
        else:  # redirected output may be readable by others: no token here
            source = ACCESS_TOKEN_ENV if os.getenv(ACCESS_TOKEN_ENV) else ACCESS_TOKEN_FILE
            print(f"Sign in at {url} with the access token from {source}.", flush=True)
    state = DashState(settings, log=lambda message: print(message, flush=True))
    decisions = state.decisions
    if settings.device_notice:
        print(settings.device_notice, flush=True)
    print(f"Decision model: {decisions.model} on {settings.device} ({decisions.why})", flush=True)

    def preload(progress=None):
        try:
            result = decisions.preload(progress)
        except Exception as exc:
            print(f"Decision model failed to load on {settings.device}: {type(exc).__name__}: {exc}", flush=True)
            raise
        print(f"Decision model loaded on {decisions.device} in {result['load_seconds']} s", flush=True)
        return result

    loading = state.jobs.submit("load", "decision model", preload, device=decisions.device)
    # The local model server (Data and models) never outlives the app. uvicorn re-raises SIGTERM after its graceful
    # exit with the handler it found; Python's default would end the process before the cleanup below.
    import atexit

    atexit.register(state.local_server.shutdown)
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    if settings.openrouter_only and state.hosted_openrouter is not None:
        print(f"LLM tasks use only OpenRouter (no local model server here): this app's hosted key for "
              f"{state.hosted_openrouter.model}, or a visitor's own API key set on Data and models.", flush=True)
    elif settings.openrouter_only:
        print("LLM tasks use only OpenRouter with each visitor's own API key (no local model server here); set it on "
              "Data and models.", flush=True)
    elif settings.local_server_only:
        print("LLM tasks use only this app's local model server (--local-server-only); start it on Data and models.",
              flush=True)
    if settings.local_server_autostart:
        threading.Thread(target=autostart, args=(state, settings.local_server_autostart, loading),
                         name="local-server-autostart", daemon=True).start()
    state.embed_cohorts()
    threading.Thread(target=state.corpus.warm_search, name="search-index", daemon=True).start()
    stop = threading.Event()
    interval = settings.rebuild_poll_seconds
    if interval > 0:  # new extraction results, a later TrialSpace round, edited grouping rules
        threading.Thread(target=state.watch, args=(interval, stop), name="watch", daemon=True).start()
    what = ("patient records and protocol content are confidential" if settings.real
            else "synthetic data only")
    print(f"Ready in {state.startup_seconds} s. Serving {settings.mode} mode on http://{settings.host}:{settings.port}/ "
          f"(research use; {what}). Ctrl-C to stop.", flush=True)
    try:
        uvicorn.run(create_app(state, access_token=token), host=settings.host, port=settings.port, log_level="warning")
    finally:
        stop.set()
        state.local_server.shutdown()  # never leave a server on the GPU
        state.encoder.close()
        state.registry.shutdown()


def autostart(state, value, loading):
    """Start the autostart spec once the decision model's load has ended, so vLLM's memory profile sees it."""
    from concurrent.futures import wait

    from .local_server import load_spec

    try:
        spec = load_spec(value)
        if loading.future is not None:
            wait([loading.future])
        state.local_server.start(spec, fill=True)  # engine, model and GPUs are enough
    except Exception as exc:
        print(f"Local model server: autostart failed: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
