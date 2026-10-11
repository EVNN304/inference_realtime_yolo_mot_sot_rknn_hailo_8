import time
from abc import ABC, abstractmethod
from threading import Thread
from typing import Optional

from src.framework.loop.io_tickable import IOTickable


class ILoop(ABC):
    def __init__(self):
        self.is_active = False

    @abstractmethod
    def on_start(self) -> None:
        """Method executed once at start of the loop"""
        ...

    @abstractmethod
    def update(self) -> None:
        """Method executed each iteration"""
        ...

    @abstractmethod
    def on_stop(self) -> None:
        """Method executed after stop is called"""
        ...


class IThreadFactory(ABC):
    @abstractmethod
    def create(self, target, name: Optional[str]) -> Thread:
        ...


class DaemonThreadFactory(IThreadFactory):
    def create(self, target, name: Optional[str]) -> Thread:
        return Thread(target=target, name=name, daemon=True)


class SimpleSeparateThreadLoopRunner:
    def __init__(
        self, loop: ILoop, thread_factory: IThreadFactory = DaemonThreadFactory()
    ) -> None:
        self._loop = loop
        self._thread_factory = thread_factory
        self._active_thread: Optional[Thread] = None

    def start(self, name: Optional[str] = None) -> None:
        self._loop.is_active = True
        self._active_thread = self._thread_factory.create(self._execute, name)
        self._active_thread.start()

    def stop(self) -> None:
        self._loop.is_active = False
        self._active_thread.join(2)
        self._active_thread = None

    def _execute(self) -> None:
        self._loop.on_start()
        while self._loop.is_active:
            self._loop.update()
        self._loop.on_stop()

    def is_running(self) -> bool:
        return self._active_thread is not None and self._active_thread.is_alive()


class TimeSyncSeparateThreadLoopRunner:
    def __init__(
        self,
        loop: ILoop,
        thread_factory: IThreadFactory = DaemonThreadFactory(),
        iter_duration: float = 1e-2,
    ) -> None:
        self._loop = loop
        self._iter_duration = iter_duration
        self._thread_factory = thread_factory
        self._clock = time.perf_counter
        self._active_thread: Optional[Thread] = None
        self._start_time = None
        self._name = None

    def start(self, name: Optional[str] = None) -> None:
        self._loop.is_active = True
        self._name = name
        self._start_time = self._clock()
        self._active_thread = self._thread_factory.create(self._execute, name)
        self._active_thread.start()

    def stop(self) -> None:
        self._loop.is_active = False
        self._active_thread.join(2)
        self._active_thread = None

    def _execute(self) -> None:
        self._loop.on_start()
        while self._loop.is_active:
            begin = self._clock()
            self._loop.update()
            end = self._clock()
            wait_time = self._iter_duration - (end - begin)
            if wait_time > 0:
                time.sleep(wait_time)
        self._loop.on_stop()

    def is_running(self) -> bool:
        return self._active_thread.is_alive()


class SimpleSameThreadLoopRunner:
    def __init__(self, loop: ILoop) -> None:
        self._loop = loop
        self._is_running = False
        self._name: Optional[str] = None

    def start(self, name: Optional[str] = None) -> None:
        if self._is_running:
            raise RuntimeError("Loop is already running.")

        self._loop.is_active = True
        self._is_running = True
        self._name = name

        try:
            self._execute()
        finally:
            self._is_running = False
            self._loop.is_active = False

    def stop(self) -> None:
        self._loop.is_active = False

    def _execute(self) -> None:
        self._loop.on_start()
        try:
            while self._loop.is_active:
                self._loop.update()
        finally:
            self._loop.on_stop()

    def is_running(self) -> bool:
        return self._is_running


class IOLoop(ILoop):
    def __init__(self, io_tickable: IOTickable) -> None:
        super().__init__()
        self._io_tickable = io_tickable

    def on_start(self) -> None:
        self._io_tickable.open()

    def update(self) -> None:
        self._io_tickable.tick_read()
        self._io_tickable.tick_write()

    def on_stop(self) -> None:
        pass
