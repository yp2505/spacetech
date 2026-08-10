"""
fsw/core/scheduler.py
---------------------
Cyclic Executive Scheduler.
Drives the main flight software loop at a deterministic frequency.
"""

import time
import logging
from fsw.core.watchdog import WatchdogManager

class CyclicScheduler:
    def __init__(self, hz: int = 10):
        if hz <= 0:
            raise ValueError("hz must be positive")
        self.hz = hz
        self.period = 1.0 / hz
        self.tasks = []
        self.watchdog = WatchdogManager(timeout_sec=self.period * 5)
        self.running = False

    def add_task(self, name: str, func, frequency_hz: int):
        """Add a task. frequency_hz must be <= self.hz and a clean divisor."""
        if frequency_hz <= 0 or frequency_hz > self.hz or self.hz % frequency_hz:
            raise ValueError("frequency_hz must be a positive divisor of scheduler hz")
        skip_ticks = self.hz // frequency_hz
        self.tasks.append({
            "name": name,
            "func": func,
            "skip": skip_ticks,
            "counter": 0
        })
        self.watchdog.register_task(name)

    def run(self, max_ticks: int = 0):
        """Run the scheduler. If max_ticks > 0, run for that many ticks then exit."""
        self.running = True
        ticks = 0
        logging.info(f"SCHEDULER: Starting cyclic executive at {self.hz} Hz.")

        while self.running:
            start_time = time.monotonic()

            for t in self.tasks:
                if t["counter"] % t["skip"] == 0:
                    try:
                        t["func"]()
                        self.watchdog.kick(t["name"])
                    except Exception as e:
                        logging.error(f"SCHEDULER: Task {t['name']} crashed: {e}")
                t["counter"] += 1

            if not self.watchdog.check():
                logging.critical("SCHEDULER: Watchdog failed. Rebooting...")
                self.running = False
                break

            ticks += 1
            if max_ticks > 0 and ticks >= max_ticks:
                self.running = False
                break

            elapsed = time.monotonic() - start_time
            sleep_time = self.period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logging.warning(f"SCHEDULER: Frame overrun by {-sleep_time:.4f}s")
