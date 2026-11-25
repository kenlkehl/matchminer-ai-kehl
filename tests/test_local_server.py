"""The local model server (``llm.local_server``) and GPU discovery (``llm.gpus``): commands, GPU tables and the
managed process, with a fake engine (no GPU, model or network)."""
from __future__ import annotations

import os
import sys
import time

import pytest

from matchminer_ai.llm import local_server as ls
from matchminer_ai.llm.gpus import GPU, parse_nvidia_smi, visible

GIB = 1024**3

# A fake OpenAI-compatible engine: serves /v1/models with its --alias on --port; --exit-after N exits with code 3;
# --ignore-term ignores SIGTERM; --child starts a worker in its process group (as engines do).
FAKE_ENGINE = r'''
import json, signal, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
args = sys.argv[1:]
value = lambda flag: args[args.index(flag) + 1]
name, port = value("--alias"), int(value("--port"))
if "--ignore-term" in args:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if "--child" in args:
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
if "--exit-after" in args:
    time.sleep(float(value("--exit-after")))
    print("loading model failed", flush=True)
    sys.exit(3)
print("main: loading model", flush=True)
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"data": [{"id": name}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a):
        pass
print("server is listening", flush=True)
HTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''


def gpu(index, physical=None, free=40, total=48, capability="8.9", name="NVIDIA L40S", source="nvml"):
    return GPU(index=index, physical_index=index if physical is None else physical, uuid=f"GPU-{index:04d}aa",
               name=name, compute_capability=capability, memory_total=total * GIB, memory_used=(total - free) * GIB,
               memory_free=free * GIB, source=source)


# ---------------------------------------------------------------------------------------------- commands


def test_vllm_command_is_loopback_with_the_gpus_tp_and_served_name():
    spec = ls.ServerSpec.from_dict({"engine": "vllm", "model": "example/unknown-model-7b", "gpus": [1, 3],
                                    "gpu_memory_utilization": 0.5})
    argv, env = ls.build_command(spec, 8123, engine_command=["vllm", "serve"],
                                 gpus=[gpu(0), gpu(1, physical=4), gpu(3, physical=6)])
    assert argv[:3] == ["vllm", "serve", "example/unknown-model-7b"]
    assert argv[argv.index("--host") + 1] == "127.0.0.1" and argv[argv.index("--port") + 1] == "8123"
    assert argv[argv.index("--served-model-name") + 1] == "example/unknown-model-7b"
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"  # default: the number of GPUs
    assert "--reasoning-parser" not in argv  # "auto" for a model no profile knows: none, not an error
    assert "--no-enable-log-requests" in argv
    # The GPUs as this process sees them, mapped to physical indices in PCI order for the child.
    assert env == {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "4,6"}


def test_llama_cpp_command_names_the_downloaded_file_and_alias():
    spec = ls.ServerSpec.from_dict({"engine": "llama_cpp", "model": "Qwen/Qwen3-0.6B-GGUF", "gguf": "q.gguf",
                                    "served_name": "Qwen/Qwen3-0.6B", "gpus": [0], "parallel": 4,
                                    "draft_model": "example/drafter-GGUF", "draft_gguf": "d.gguf",
                                    "draft_type": "draft-dflash", "draft_tokens": 6})
    argv, env = ls.build_command(spec, 9000, engine_command=["llama-server"], files={"gguf": "/c/q.gguf",
                                                                                    "draft": "/c/d.gguf"},
                                 gpus=[gpu(0, physical=2)])
    assert argv[argv.index("--model") + 1] == "/c/q.gguf" and argv[argv.index("--alias") + 1] == "Qwen/Qwen3-0.6B"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert {"--kv-unified", "--jinja", "--no-webui"} <= set(argv)
    assert argv[argv.index("--spec-draft-model") + 1] == "/c/d.gguf"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"


def test_spec_refuses_unknown_fields_and_address_flags():
    with pytest.raises(ValueError, match="Unknown"):
        ls.ServerSpec.from_dict({"engine": "vllm", "model": "a/b", "gpus": [0], "hostname": "x"})
    for arg in ("--host", "--port=1", "--api-key"):
        with pytest.raises(ValueError, match="set by the app"):
            ls.ServerSpec.from_dict({"engine": "llama_cpp", "model": "a/b", "extra_args": [arg, "0.0.0.0"]})
    with pytest.raises(ValueError, match="cannot be set"):
        ls.ServerSpec.from_dict({"engine": "llama_cpp", "model": "a/b", "env": {"LLAMA_ARG_HOST": "0.0.0.0"}})


def test_bolt_proposes_llama_cpp_on_ada_and_vllm_on_blackwell():
    installed = {"vllm": {"available": True}, "llama_cpp": {"available": True}}
    model = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"
    ada = ls.propose_engine(model, [gpu(0)], installed)
    assert ada["engine"] == "llama_cpp" and "Blackwell" in ada["reason"]
    assert ls.propose_engine(model, [gpu(0, capability="12.0")], installed)["engine"] == "vllm"


# ---------------------------------------------------------------------------------------------- GPUs


def test_nvidia_smi_tables_and_visible_remapping():
    gpus = parse_nvidia_smi("0, GPU-aaa, NVIDIA H100, 9.0, 81920, 1024, 80896, 3\n"
                            "1, GPU-bbb, NVIDIA H100, 9.0, 81920, [N/A], [N/A], [N/A]\n",
                            "GPU-aaa, 4242, 1000\nGPU-aaa, 4343, [N/A]\n")
    assert [g.physical_index for g in gpus] == [0, 1]
    assert gpus[0].memory_free == 80896 * 1024**2 and gpus[0].compute_capability == "9.0"
    assert [(p.pid, p.used) for p in gpus[0].processes] == [(4242, 1000 * 1024**2), (4343, None)]
    assert gpus[1].memory_used is None
    # CUDA_VISIBLE_DEVICES=1 or its UUID: the second GPU, seen as GPU 0.
    for value in ("1", "GPU-bbb"):
        (only,) = visible(gpus, value)
        assert (only.index, only.physical_index) == (0, 1)
    assert [g.physical_index for g in visible(gpus, "1,0")] == [1, 0]
    assert visible(gpus, "") == []


# ---------------------------------------------------------------------------------------------- the process


@pytest.fixture
def fake_engine(tmp_path):
    script = tmp_path / "fake_engine.py"
    script.write_text(FAKE_ENGINE)
    return [sys.executable, str(script)]


def server(tmp_path, fake_engine, *extra):
    spec = ls.ServerSpec.from_dict({"engine": "llama_cpp", "model": "example/fake-GGUF", "served_name": "fake",
                                    "gpus": [], "extra_args": list(extra)})
    return ls.ManagedServer(spec, tmp_path / "logs", engine_command=fake_engine, download=False, poll_seconds=0.05)


def test_server_is_ready_when_models_lists_it_and_stop_kills_its_group(tmp_path, fake_engine):
    ended = []
    s = server(tmp_path, fake_engine, "--child", "--ignore-term")
    s.on_end = ended.append
    s.start()
    assert s.wait(timeout=30) == "ready", s.log_tail()
    pgid = s.process.pid
    assert ls.group_alive(pgid)
    started = time.monotonic()
    s.stop(grace=0.5)  # the leader ignores SIGTERM: SIGKILL ends the group, its worker too
    while ls.group_alive(pgid) and time.monotonic() - started < 10:  # an orphaned worker's zombie, until reaped
        time.sleep(0.05)
    assert s.state == "stopped" and not ls.group_alive(pgid)
    assert time.monotonic() - started < 15 and ended == [s]
    assert s.log_tail()[0].startswith("$ ")  # the log begins with the command line


def test_an_engine_that_exits_is_reported_with_its_code(tmp_path, fake_engine):
    s = server(tmp_path, fake_engine, "--exit-after", "0.1").start()
    assert s.wait(timeout=30) == "exited"
    assert s.exit_code == 3 and not s.active
    s.stop()  # nothing to stop
    assert s.state == "exited"


def test_download_failure_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(ls, "plan_downloads", lambda spec: (_ for _ in ()).throw(LookupError("not on the Hub")))
    spec = ls.ServerSpec.from_dict({"engine": "llama_cpp", "model": "example/missing", "gpus": []})
    s = ls.ManagedServer(spec, tmp_path / "logs", engine_command=["llama-server"]).start()
    assert s.wait(timeout=10) == "failed" and "not on the Hub" in s.error
    assert os.path.exists(s.log_path)
