"""arm_driver: the only process that touches the follower's servo bus.

    python -m yogurt.arm_driver [--backend sim|real] [--config PATH]

Subscribes arm/target, system/stop, system/resume; publishes arm/state every tick. Every command
passes through yogurt.safety.SafetyFilter.
"""
