# -*- coding: utf-8 -*-
"""Command-line interface: doctor / bench / train / serve."""
import argparse
import json
import sys


def _build_parser():
    p = argparse.ArgumentParser(prog="tevforge",
                                description="Long-horizon agent dense-reward RL training framework")
    p.add_argument("--config", default=None, help="path to a YAML config")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("doctor", help="smoke-test every component")
    sub.add_parser("bench", help="run the benchmark suite")
    sub.add_parser("serve", help="start the stdlib HTTP server")

    t = sub.add_parser("train", help="train a policy with PPRS + GRPO")
    t.add_argument("--steps", type=int, default=40)
    t.add_argument("--tasks", type=int, default=8)
    return p


def cmd_doctor(cfg):
    from ..world.answer_world import AnswerWorld
    from ..synth.gym import TevGym
    from ..reward.pprs import PPRS
    from ..agent.policy import MockPolicy
    from ..agent.rollout import rollout
    from ..model.backends import list_backends
    from ..engine.train_engine import TrainEngine

    world = AnswerWorld(seed=cfg.world.seed)
    gym = TevGym(world=world)
    tasks = gym.build(tasks_per_domain=1)
    policy = MockPolicy(skill=cfg.model.skill)
    traj = rollout(policy, world, tasks[0], PPRS().step_reward)

    engine = TrainEngine(cfg)
    engine.fit(n_tasks=8, max_steps=2)

    out = {
        "backends": list_backends(),
        "domains": gym.domains,
        "action_space": len(world.actions),
        "horizon_range": [world.horizon_min, world.horizon_max],
        "sample_task": tasks[0],
        "sample_traj_len": traj.length,
        "sample_progress": traj.final_progress,
        "declared": cfg.declared.to_dict(),
        "engine_smoke": "ok",
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


def cmd_bench(cfg):
    from ..bench.benchmarks import run_all
    report = run_all(cfg)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def cmd_train(cfg, steps, tasks):
    from ..engine.train_engine import TrainEngine
    engine = TrainEngine(cfg)
    result = engine.fit(n_tasks=tasks, max_steps=steps)
    print(json.dumps({
        "final_skill": result["final_skill"],
        "iterations": len(result["history"]),
        "first": result["history"][0] if result["history"] else None,
        "last": result["history"][-1] if result["history"] else None,
    }, ensure_ascii=False, indent=2))
    return result


def cmd_serve(cfg):
    from .http import make_server
    server = make_server(cfg)
    host, port = "127.0.0.1", 8090
    print("tevforge serving on http://%s:%d" % (host, port))
    server.serve_forever()


def main(argv=None):
    from ..config import load_config
    args = _build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd == "doctor":
        cmd_doctor(cfg)
    elif args.cmd == "bench":
        cmd_bench(cfg)
    elif args.cmd == "train":
        cmd_train(cfg, args.steps, args.tasks)
    elif args.cmd == "serve":
        cmd_serve(cfg)
    else:
        _build_parser().print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
