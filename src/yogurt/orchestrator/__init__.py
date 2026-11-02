"""orchestrator: the fetch-and-handoff state machine (see fsm.py).

    python -m yogurt.orchestrator [--dev] [--config PATH]

Turns bus messages into events, feeds them to the state machine, and publishes what it returns.
In dev mode (config orchestrator.dev_mode or --dev), `python -m yogurt.devkeys` keypresses on
dev/key stand in for the wake word, ASR, vision, and the policy.
"""
