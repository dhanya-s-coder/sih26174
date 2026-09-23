import time
from typing import Dict, List, Tuple
from src.har_space.data_models import HandState, Detection, Interaction, Event
from src.har_space.events.bus import EventBus

class InteractionExtractor:
    def __init__(self, event_bus: EventBus, padding: float = 0.05, min_frames: int = 5, release_frames: int = 3):
        self.bus = event_bus
        self.padding = padding
        self.min_frames = min_frames
        self.release_frames = release_frames
        
        # State: { (hand_idx, obj_label): {"frames": int, "active": bool, "start_time": float} }
        self.contact_state = {}

    def process(self, hands: List[HandState], objects: List[Detection], video_ts: float):
        current_contacts = set()
        
        for h_idx, hand in enumerate(hands):
            for obj in objects:
                if self._check_contact(hand, obj):
                    key = (h_idx, obj.label)
                    current_contacts.add(key)
                    
                    if key not in self.contact_state:
                        self.contact_state[key] = {"frames": 0, "active": False, "start_time": video_ts, "release_count": 0}
                    
                    state = self.contact_state[key]
                    state["frames"] += 1
                    state["release_count"] = 0
                    
                    if state["frames"] >= self.min_frames and not state["active"]:
                        state["active"] = True
                        state["start_time"] = video_ts
                        self.bus.publish(Event(event_type="interaction", timestamp=video_ts, payload={
                            "subject": "hand", "verb": "touches", "object": obj.label
                        }))
                    elif state["active"]:
                        duration = video_ts - state["start_time"]
                        if duration >= 1.5:
                            self.bus.publish(Event(event_type="interaction", timestamp=video_ts, payload={
                                "subject": "hand", "verb": "holds", "object": obj.label
                            }))
                            
        # Handle releases
        for key, state in list(self.contact_state.items()):
            if key not in current_contacts:
                state["release_count"] += 1
                if state["release_count"] >= self.release_frames:
                    if state["active"]:
                        self.bus.publish(Event(event_type="interaction", timestamp=video_ts, payload={
                            "subject": "hand", "verb": "releases", "object": key[1]
                        }))
                    del self.contact_state[key]

    def _check_contact(self, hand: HandState, obj: Detection) -> bool:
        ox1, oy1, ox2, oy2 = obj.bbox
        # Add padding
        ox1 -= self.padding; oy1 -= self.padding
        ox2 += self.padding; oy2 += self.padding
        
        # A hand touches if any landmark is inside the padded box
        for (x, y, z) in hand.landmarks:
            if ox1 <= x <= ox2 and oy1 <= y <= oy2:
                return True
        return False
