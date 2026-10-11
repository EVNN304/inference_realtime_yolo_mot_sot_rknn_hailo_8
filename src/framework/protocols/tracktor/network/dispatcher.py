from typing import List, Optional

from src.framework.protocols.tracktor.network.handlers import Handler


class Dispatcher:
    def __init__(self, handlers: Optional[List[Handler]] = None):
        self._handlers: dict[int, Handler] = {}
        if handlers is not None:
            for handler in handlers:
                self.add_handler(handler)

    def add_handler(self, handler) -> None:
        self._handlers[handler.msg_id()] = handler

    def dispatch(self, message) -> None:
        handler = self._handlers.get(message.msg_id(), None)
        if handler is not None:
            handler(message)
