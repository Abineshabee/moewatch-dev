# =============================================================================
#
# ╔╦╗ ╔═╗ ╔═╗ ╦ ╦ ╔═╗ ╔╦╗ ╔═╗ ╦ ╦
# ║║║ ║ ║ ║╣  ║║║ ╠═╣  ║  ║   ╠═╣
# ╩ ╩ ╚═╝ ╚═╝ ╚╩╝ ╩ ╩  ╩  ╚═╝ ╩ ╩  v0.2.0
#
# moewatch/collector/ring_buffer.py
# =============================================================================
#
# Project      : MoEWatch
# Version      : v0.2.0
# Description  : Fixed-memory circular buffer for event streams. Used by
#                 StatCollector to store RoutingEvent and GradientEvent
#                 objects per layer (and per expert) with a bounded
#                 capacity, so long training runs do not grow memory
#                 usage unboundedly.
#
#                 Thread-safe write and read paths via a single
#                 threading.Lock per buffer instance. Supports both
#                 windowed reads (most recent N events — used by CUSUM
#                 and rolling-statistics analyzers) and full reads.
#
# Author       : Abinesh N (@Abineshabee)
# Repository   : https://github.com/Abineshabee/MoEWatch
# License      : Apache 2.0
#
# Contents
# --------
#   RingBuffer — generic fixed-capacity circular buffer of events
#
# Usage
# -----
#   from moewatch.collector.ring_buffer import RingBuffer
#
#   buf: RingBuffer[RoutingEvent] = RingBuffer(capacity=1000)
#   buf.write(event)
#   recent = buf.read_window(50)
#   everything = buf.read_all()
#
# =============================================================================

from __future__ import annotations

import threading
from collections import deque
from typing import Generic, List, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """Fixed-capacity, thread-safe circular buffer of events.

    Backed by ``collections.deque`` with an explicit ``maxlen``, which
    PyTorch and the standard library both implement as an O(1) circular
    buffer: once ``capacity`` items are present, appending a new item
    automatically and atomically drops the oldest one.

    All public methods acquire an internal :class:`threading.Lock`,
    making this class safe to write from hook callbacks (which may fire
    on background CUDA streams or from PyTorch's autograd engine) while
    being read concurrently from the main monitoring loop.

    Parameters
    ----------
    capacity : int, optional
        Maximum number of events to retain. Must be a positive integer.
        Default: ``1000``.

    Attributes
    ----------
    capacity : int
        Maximum number of events this buffer will retain.

    Raises
    ------
    ValueError
        If ``capacity`` is not a positive integer.

    Notes
    -----
    ``RingBuffer`` is intentionally type-parameterized (``Generic[T]``)
    but performs no runtime type checking on ``write()`` — callers are
    responsible for writing a consistent event type per buffer instance
    (e.g. one buffer per layer for :class:`RoutingEvent`, one buffer per
    ``(layer, expert)`` pair for :class:`GradientEvent`).
    """

    def __init__(self, capacity: int = 1000) -> None:
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(
                f"[MoEWatch] RingBuffer: capacity must be a positive "
                f"integer, got {capacity!r}."
            )

        self.capacity: int = capacity
        self._buffer: "deque[T]" = deque(maxlen=capacity)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def write(self, event: T) -> None:
        """Append an event to the buffer, dropping the oldest if full.

        Parameters
        ----------
        event : T
            The event object to store (e.g. a :class:`RoutingEvent` or
            :class:`GradientEvent`).

        Returns
        -------
        None

        Notes
        -----
        Thread-safe. If the buffer is already at :attr:`capacity`, the
        oldest event is silently discarded (this is the ``deque``'s
        built-in ``maxlen`` behavior — no explicit ``popleft()`` is
        needed).
        """
        with self._lock:
            self._buffer.append(event)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def read_window(self, window_size: int) -> List[T]:
        """Return the most recent ``window_size`` events.

        Parameters
        ----------
        window_size : int
            Number of most-recent events to return. If
            ``window_size <= 0``, an empty list is returned. If
            ``window_size`` exceeds the number of events currently
            stored, all stored events are returned (oldest-to-newest).

        Returns
        -------
        list[T]
            A new list containing up to ``window_size`` of the most
            recently written events, in chronological order (oldest
            first, newest last). The returned list is a copy; mutating
            it does not affect the buffer.

        Notes
        -----
        Thread-safe read: the entire slice is taken while holding the
        lock to guarantee a consistent snapshot even if a concurrent
        writer is active.
        """
        if window_size <= 0:
            return []

        with self._lock:
            if window_size >= len(self._buffer):
                return list(self._buffer)
            # deque supports negative-index slicing via islice-like
            # access; convert to list first for a simple, correct slice.
            return list(self._buffer)[-window_size:]

    def read_all(self) -> List[T]:
        """Return a copy of every event currently in the buffer.

        Returns
        -------
        list[T]
            All stored events in chronological order (oldest first,
            newest last). The returned list is a copy; mutating it does
            not affect the buffer.

        Notes
        -----
        Thread-safe read.
        """
        with self._lock:
            return list(self._buffer)

    def length(self) -> int:
        """Return the current number of events stored.

        Returns
        -------
        int
            Number of events currently in the buffer. Always
            ``0 <= length() <= capacity``.

        Notes
        -----
        Thread-safe read.
        """
        with self._lock:
            return len(self._buffer)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all events from the buffer.

        Returns
        -------
        None

        Notes
        -----
        Thread-safe. After this call, :meth:`length` returns ``0`` and
        :meth:`read_all` / :meth:`read_window` return empty lists until
        new events are written.
        """
        with self._lock:
            self._buffer.clear()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Alias for :meth:`length`, enabling ``len(buffer)``.

        Returns
        -------
        int
            Number of events currently stored.
        """
        return self.length()

    def __repr__(self) -> str:
        return (
            f"RingBuffer(capacity={self.capacity}, "
            f"length={self.length()})"
        )
