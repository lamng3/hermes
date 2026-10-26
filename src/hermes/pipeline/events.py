"""Observers of a pipeline run: each stage reports when it starts, ends, or fails."""

import sys
from dataclasses import dataclass, field


@dataclass
class Event:
    stage: str
    kind: str  # start, end, error
    data: dict = field(default_factory=dict)


class Observer:
    def on_event(self, event):
        raise NotImplementedError


class Recorder(Observer):
    """Keeps every event, for tests, the CLI and the demo."""

    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)

    def ended(self, stage):
        return next((e.data for e in self.events if e.stage == stage and e.kind == "end"), None)


class Printer(Observer):
    """Prints one line per finished stage."""

    def __init__(self, stream=sys.stderr):
        self.stream = stream

    def on_event(self, event):
        if event.kind == "start":
            return
        shown = {k: v for k, v in event.data.items() if k not in ("prompt", "reply", "spend")}
        if "spend" in event.data:
            shown["spent"] = event.data["spend"]["total"]
        print(f"[{event.stage}] {event.kind}: {shown}", file=self.stream)
