import logging
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar, final

TMessage = TypeVar("TMessage")


class Handler(ABC, Generic[TMessage]):
    def __init__(self, msg_type: TMessage):
        self._msg_type = msg_type
        self._logger = logging.getLogger(msg_type.PROTOCOL)

    @final
    def msg_id(self) -> int:
        return self._msg_type.msg_id()

    @property
    @final
    def msg_type(self) -> TMessage:
        return self._msg_type

    def __call__(self, msg: TMessage) -> None:
        if isinstance(msg, self._msg_type):
            try:
                self.handle(msg)
            except Exception as e:
                self._logger.error(
                    "Error '%s' occuried while handling message with msg_id=%d",
                    str(e),
                    self.msg_id()
                )
        else:
            self._logger.warning(
                "Wrong type received for msg_id=%d. Expected %s, received %s",
                self.msg_id(),
                str(self._msg_type),
                str(type(msg)),
            )

    @abstractmethod
    def handle(self, msg: TMessage) -> None: ...


class HandlerList(Handler):
    def __init__(self, msg_type: TMessage, handlers: List[Handler] = []):
        super().__init__(msg_type)
        self._handlers = handlers

    def handle(self, msg: TMessage):
        for handler in self._handlers:
            handler.handle(msg)

    def append(self, handler: TMessage):
        if handler.msg_type == self._msg_type:
            self._handlers.append(handler)
        else:
            self._logger.warning("Wrong type of handler for handler list")
