"""
fsw/core/watchdog.py
--------------------
Watchdog timer to detect frozen tasks or unresponsive subsystems.
"""

import time
import logging

class WatchdogManager:
    def __init__(self, timeout_sec: float = 2.0):
        self.timeout_sec = timeout_sec
        self._last_kicks = {}

    def register_task(self, task_name: str):
        """Register a task to be monitored."""
        self._last_kicks[task_name] = time.time()

    def kick(self, task_name: str):
        """Called by a task to prove it is still alive (patting the dog)."""
        if task_name in self._last_kicks:
            self._last_kicks[task_name] = time.time()

    def check(self) -> bool:
        """Check all tasks. Returns True if all are alive, False if any hung."""
        now = time.time()
        for task, last_kick in self._last_kicks.items():
            if (now - last_kick) > self.timeout_sec:
                logging.error(f"WATCHDOG TIMEOUT: Task '{task}' hung!")
                return False
        return True
