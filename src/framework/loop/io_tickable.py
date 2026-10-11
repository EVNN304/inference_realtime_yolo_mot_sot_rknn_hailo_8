from abc import ABC, abstractmethod


class IOTickable(ABC):
    @abstractmethod
    def open(self) -> None:
        ...

    @abstractmethod
    def tick_read(self) -> None:
        ...

    @abstractmethod
    def tick_write(self) -> None:
        ...
