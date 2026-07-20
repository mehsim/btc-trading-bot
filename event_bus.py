import asyncio
import time

class EventType:
    CANDLE_RECEIVED = "CandleReceived"
    FEATURES_COMPUTED = "FeaturesComputed"
    SIGNAL_GENERATED = "SignalGenerated"
    KILL_SWITCH_TRIGGERED = "KillSwitchTriggered"

class Event:
    def __init__(self, event_type: str, data: dict):
        self.event_type = event_type
        self.data = data
        self.timestamp = time.time()

class AsyncEventBus:
    def __init__(self):
        self.queue = asyncio.Queue()
        self._handlers = {}

    def subscribe(self, event_type: str, handler):
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    async def publish(self, event_type: str, data: dict):
        event = Event(event_type, data)
        await self.queue.put(event)

    async def start_processing(self):
        while True:
            event = await self.queue.get()
            handlers = self._handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    print(f"[EventBus Error] Exception handling event {event.event_type}: {e}")
            self.queue.task_done()

global_event_bus = AsyncEventBus()
