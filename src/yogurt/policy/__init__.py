"""policy: runs arm motions for the orchestrator and publishes arm/target.

    python -m yogurt.policy [--backend replay|act] [--config PATH]

Subscribes policy/command and arm/state; publishes arm/target while a task runs, and
policy/status (running, done, failed, stopped). Tasks: grasp, handoff, release_home.

Backends: replay (taught poses, data/poses/<task>.yaml) and act (TODO, act.py).
"""
