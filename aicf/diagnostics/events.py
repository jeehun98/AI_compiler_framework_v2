from __future__ import annotations
from typing import Callable

_LISTENERS: list[Callable[[str, object], None]] = []


def add_listener(listener):
    _LISTENERS.append(listener)


def clear_listeners():
    _LISTENERS.clear()


def emit(context, event: str, payload):
    if not getattr(context, "diagnostics", False):
        return
    for listener in list(_LISTENERS):
        listener(event, payload)
