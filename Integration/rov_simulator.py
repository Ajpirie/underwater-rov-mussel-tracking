"""Deterministic fake ROV feedback for handover and offline state-machine tests."""
from __future__ import annotations

import heapq


class SimulatedROV:
    """Acknowledge requests and report timed arrival/completion; never actuates hardware."""

    def __init__(self, ack_delay_s=0.1, approach_duration_s=2.0,
                 collection_duration_s=1.0, accept_commands=True,
                 report_arrival=True, report_completion=True):
        self.ack_delay_s = float(ack_delay_s)
        self.approach_duration_s = float(approach_duration_s)
        self.collection_duration_s = float(collection_duration_s)
        self.accept_commands = accept_commands
        self.report_arrival = report_arrival
        self.report_completion = report_completion
        self._pending = []
        self._sequence = 0
        self.commands = []

    def _schedule(self, when, event):
        heapq.heappush(self._pending, (when, self._sequence, event))
        self._sequence += 1

    def request_approach(self, target, now_s):
        self.commands.append(('approach', tuple(target)))
        if not self.accept_commands:
            self._schedule(now_s + self.ack_delay_s, 'rejected')
            return
        self._schedule(now_s + self.ack_delay_s, 'approach_ack')
        if self.report_arrival:
            self._schedule(now_s + max(self.ack_delay_s, self.approach_duration_s), 'arrived')

    def request_collection(self, target, now_s):
        self.commands.append(('collect', tuple(target)))
        if not self.accept_commands:
            self._schedule(now_s + self.ack_delay_s, 'rejected')
            return
        self._schedule(now_s + self.ack_delay_s, 'collection_ack')
        if self.report_completion:
            self._schedule(now_s + max(self.ack_delay_s, self.collection_duration_s), 'complete')

    def poll(self, now_s):
        events = []
        while self._pending and self._pending[0][0] <= now_s:
            events.append(heapq.heappop(self._pending)[2])
        return events

    def safe_stop(self):
        """Clear simulated work; this is not a physical stop or E-stop."""
        self._pending.clear()
        self.commands.append(('stop', None))
