# -*- coding: utf-8 -*-
"""A dependency-free stdlib HTTP server exposing the training engine.

Endpoints:
  GET /health           -> {"status": "ok"}
  GET /v1/models        -> declared model metadata
  POST /v1/rollout      -> {"task_id": ..., "skill": ...} -> trajectory summary
  POST /v1/bench        -> benchmark report
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


def make_handler(cfg):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._json({"status": "ok", "service": "tevforge"})
            elif self.path == "/v1/models":
                self._json({"model_id": cfg.model.model_id,
                            "backend": cfg.model.backend,
                            "declared": cfg.declared.to_dict()})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            from ..world.answer_world import AnswerWorld
            from ..agent.policy import MockPolicy
            from ..agent.rollout import rollout
            from ..reward.pprs import PPRS
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception:
                data = {}
            world = AnswerWorld(seed=cfg.world.seed)
            if self.path == "/v1/rollout":
                tid = data.get("task_id", "api:0")
                skill = float(data.get("skill", cfg.model.skill))
                policy = MockPolicy(skill=skill)
                traj = rollout(policy, world, tid, PPRS().step_reward)
                self._json({"task_id": tid, "length": traj.length,
                            "progress": traj.final_progress, "completed": traj.completed})
            elif self.path == "/v1/bench":
                from ..bench.benchmarks import run_all
                self._json(run_all(cfg))
            else:
                self._json({"error": "not found"}, 404)

        def log_message(self, *a):
            pass  # keep CI output clean

    return Handler


def make_server(cfg):
    return HTTPServer(("127.0.0.1", 8090), make_handler(cfg))
