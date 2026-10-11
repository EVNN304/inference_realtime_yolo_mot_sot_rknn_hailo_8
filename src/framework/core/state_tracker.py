import time
from typing import Generic, TypeVar

T = TypeVar('T')

class StateTracker(Generic[T]):
    __slots__ = ("_previous", "_current")

    def __init__(self, previous: T = None, current: T = None) -> None:
        self._previous: T = previous
        self._current: T = current

    def update(self, new_state: T) -> None:
        """Updates state of the parameter."""
        self._previous = self._current
        self._current = new_state

    def has_changed(self) -> bool:
        """Returns True if current state is different from previous state."""
        return self._current != self._previous

    def get(self) -> T:
        """Returns current state."""
        return self._current

    def __call__(self) -> T:
        """Allows the instance to be called like a function to get the current state."""
        return self._current

    def get_previous(self) -> T:
        """Returns previous state."""
        return self._previous


class TimeFadingStateTracker(Generic[T]):
    __slots__ = ("_tracker", "changing_timeout", "_latest_update")

    def __init__(self, changing_timeout: float = 100e-3, previous: T = None, current: T = None) -> None:
        self._tracker = StateTracker[T](previous, current)
        self.changing_timeout = changing_timeout
        self._latest_update = 0.0

    def update(self, new_state: T) -> None:
        """Updates state of the parameter."""
        self._tracker.update(new_state)
        if self._tracker.has_changed():
            self._latest_update = time.time()

    def has_changed(self) -> bool:
        return self._tracker.has_changed()

    def is_changing(self) -> bool:
        now = time.time()
        return now <= self._latest_update + self.changing_timeout

    def get(self) -> T:
        return self._tracker.get()

    def get_previous(self) -> T:
        return self._tracker.get_previous()
