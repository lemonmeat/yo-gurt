# Yo-gurt

A voice- and vision-controlled SO-101 robot arm that fetches a yogurt cup and hands it to the
person who asked. Course final project, fall 2026.

```bash
uv sync                      # core + dev tools (no torch)
scripts/fetch_so_arm100.sh   # MuJoCo model + URDF for sim and forward kinematics
uv run pytest
uv run honcho start          # whole stack, follower in MuJoCo sim
```

See [CLAUDE.md](CLAUDE.md) for the architecture and safety rules, [docs/PLAN.md](docs/PLAN.md) for
the schedule, and [docs/TASKS.md](docs/TASKS.md) for the backlog.
