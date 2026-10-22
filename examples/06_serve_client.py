# -*- coding: utf-8 -*-
"""Talk to the stdlib HTTP server (start it first with `python -m tevforge serve`).

Run (two terminals):
    python -m tevforge serve
    python examples/06_serve_client.py
"""
import json
import urllib.request

base = "http://127.0.0.1:8090"


def get(path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def post(path, payload):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


print("health :", get("/health"))
print("models :", get("/v1/models"))
print("rollout:", post("/v1/rollout", {"task_id": "api:1", "skill": 0.9}))
