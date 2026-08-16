"""
fsw/core/watchdog.py
--------------------
Watchdog timer with heartbeat mechanism for AI Brain and subsystem monitoring.
"""

import time
import logging
import threading
from typing import Callable, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum


class WatchdogAction(Enum):
    """Actions to take on watchdog timeout."""
    LOG_ONLY = "log_only"
    TRIGGER_SAFE_MODE = "trigger_safe_mode"
    RESET_AI_BRAIN = "reset_ai_brain"
    FULL_SYSTEM_RESET = "full_system_reset"


@dataclass
class WatchdogTask:
    """Configuration for a monitored task."""
    name: str
    timeout_sec: float
    action: WatchdogAction = WatchdogAction.LOG_ONLY
    callback: Optional[Callable] = None
    last_kick: float = field(default_factory=time.time)
    enabled: bool = True


class WatchdogManager:
    """
    Watchdog Manager with heartbeat support for AI Brain.
    
    Features:
    - Per-task configurable timeouts and actions
    - Heartbeat monitoring for AI Brain inference
    - Automatic safe mode trigger on critical task failure
    - Thread-safe operation
    """
    
    def __init__(self, default_timeout_sec: float = 2.0):
        self.default_timeout_sec = default_timeout_sec
        self._tasks: Dict[str, WatchdogTask] = {}
        self._lock = threading.RLock()
        self._callbacks: Dict[WatchdogAction, Callable] = {}
        self._register_default_callbacks()
    
    def _register_default_callbacks(self):
        self._callbacks[WatchdogAction.LOG_ONLY] = lambda task: logging.error(f"WATCHDOG TIMEOUT: Task '{task.name}' hung!")
        self._callbacks[WatchdogAction.TRIGGER_SAFE_MODE] = lambda task: logging.critical(f"WATCHDOG: Triggering SAFE_MODE due to '{task.name}' timeout")
        self._callbacks[WatchdogAction.RESET_AI_BRAIN] = lambda task: logging.critical(f"WATCHDOG: Resetting AI Brain due to '{task.name}' timeout")
        self._callbacks[WatchdogAction.FULL_SYSTEM_RESET] = lambda task: logging.critical(f"WATCHDOG: Full system reset due to '{task.name}' timeout")
    
    def register_task(self, name: str, timeout_sec: float = None, 
                      action: WatchdogAction = WatchdogAction.LOG_ONLY,
                      callback: Optional[Callable] = None) -> None:
        """Register a task to be monitored."""
        with self._lock:
            self._tasks[name] = WatchdogTask(
                name=name,
                timeout_sec=timeout_sec or self.default_timeout_sec,
                action=action,
                callback=callback,
                last_kick=time.time(),
                enabled=True
            )
            logging.info(f"WATCHDOG: Registered task '{name}' (timeout={self._tasks[name].timeout_sec}s, action={action.value})")
    
    def register_ai_heartbeat(self, timeout_sec: float = 5.0) -> None:
        """Register AI Brain heartbeat with safe mode trigger on timeout."""
        self.register_task(
            "AI_BRAIN_HEARTBEAT",
            timeout_sec=timeout_sec,
            action=WatchdogAction.TRIGGER_SAFE_MODE
        )
    
    def kick(self, task_name: str) -> bool:
        """Called by a task to prove it is still alive. Returns True if task exists."""
        with self._lock:
            if task_name in self._tasks:
                self._tasks[task_name].last_kick = time.time()
                return True
            return False
    
    def ai_heartbeat(self) -> None:
        """AI Brain calls this after each successful inference."""
        self.kick("AI_BRAIN_HEARTBEAT")
    
    def check(self) -> Dict[str, bool]:
        """
        Check all tasks. Returns dict of task_name -> is_alive.
        Executes callbacks for timed-out tasks.
        """
        with self._lock:
            now = time.time()
            results = {}
            for name, task in self._tasks.items():
                if not task.enabled:
                    results[name] = True
                    continue
                
                elapsed = now - task.last_kick
                is_alive = elapsed <= task.timeout_sec
                results[name] = is_alive
                
                if not is_alive:
                    logging.error(f"WATCHDOG TIMEOUT: Task '{name}' hung! (elapsed={elapsed:.1f}s, timeout={task.timeout_sec}s)")
                    self._execute_action(task)
            
            return results
    
    def _execute_action(self, task: WatchdogTask):
        """Execute the configured action for a timed-out task."""
        if task.action in self._callbacks:
            try:
                self._callbacks[task.action](task)
            except Exception as e:
                logging.error(f"WATCHDOG callback failed for '{task.name}': {e}")
        
        if task.callback:
            try:
                task.callback()
            except Exception as e:
                logging.error(f"WATCHDOG custom callback failed for '{task.name}': {e}")
    
    def disable_task(self, name: str) -> None:
        """Temporarily disable monitoring for a task."""
        with self._lock:
            if name in self._tasks:
                self._tasks[name].enabled = False
    
    def enable_task(self, name: str) -> None:
        """Re-enable monitoring for a task."""
        with self._lock:
            if name in self._tasks:
                self._tasks[name].enabled = True
                self._tasks[name].last_kick = time.time()
    
    def get_task_status(self, name: str) -> Optional[Dict]:
        """Get detailed status for a task."""
        with self._lock:
            if name not in self._tasks:
                return None
            task = self._tasks[name]
            now = time.time()
            return {
                "name": name,
                "enabled": task.enabled,
                "timeout_sec": task.timeout_sec,
                "action": task.action.value,
                "elapsed_sec": now - task.last_kick,
                "is_alive": (now - task.last_kick) <= task.timeout_sec
            }
    
    def get_all_status(self) -> Dict[str, Dict]:
        """Get status for all tasks."""
        with self._lock:
            return {name: self.get_task_status(name) for name in self._tasks}
