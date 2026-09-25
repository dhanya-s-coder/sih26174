from typing import Callable, Dict, List
from src.har_space.data_models import Event

class EventBus:
    """A simple in-process event bus for module communication."""
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = {}
    
    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        """Subscribes a handler to a specific event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        
    def publish(self, event: Event) -> None:
        """Publishes an event to all subscribed handlers."""
        handlers = self._subscribers.get(event.event_type, [])
        for handler in handlers:
            handler(event)
