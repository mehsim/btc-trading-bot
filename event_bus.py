"""
Asynchronous Decoupled Event Bus Architecture.
Dispatches SignalEvent, RiskApprovedEvent, ExecutionEvent, and StateChangeEvent across system modules.
"""

from typing import Dict, Any, List, Callable
from collections import defaultdict

class EventBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = defaultdict(list)

    def subscribe(self, event_type: str, callback: Callable[[Dict[str, Any]], None]):
        self.subscribers[event_type].append(callback)

    def publish(self, event_type: str, data: Dict[str, Any]):
        handlers = self.subscribers.get(event_type, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                print(f"[EventBus Error] Event '{event_type}' handler failed: {e}")

event_bus = EventBus()
