"""Minimal supervisory ROV sequence; independent of tracking and vehicle hardware."""
from __future__ import annotations

from enum import Enum
import math
import time


class ROVState(str, Enum):
    IDLE = 'IDLE'
    APPROACH = 'APPROACH'
    COLLECT = 'COLLECT'
    FAULT = 'FAULT'


class ROVController:
    """Operator-gated approach and collection with feedback and timeouts."""

    def __init__(self, adapter, ack_timeout_s=1.0, action_timeout_s=15.0):
        if not (0 < ack_timeout_s < action_timeout_s):
            raise ValueError('Require 0 < acknowledgement timeout < action timeout')
        self.adapter = adapter
        self.ack_timeout_s = float(ack_timeout_s)
        self.action_timeout_s = float(action_timeout_s)
        self.state = ROVState.IDLE
        self.armed = False
        self.arrived = False
        self.acknowledged = False
        self.target = None
        self.fault_reason = None
        self.started_at = None
        self.history = []

    def arm(self):
        """Allow an operator-initiated approach while idle."""
        if self.state != ROVState.IDLE:
            raise RuntimeError('Reset or complete the current sequence first')
        self.armed = True
        self.history.append('Simulation armed')

    def start_approach(self, target, now_s=None):
        """Send a simulated approach request for a frozen camera-relative target."""
        if self.state != ROVState.IDLE or not self.armed:
            raise RuntimeError('Arm the idle controller before approaching')
        try:
            target = tuple(float(v) for v in target)
        except (TypeError, ValueError) as exc:
            raise ValueError('Target must contain three finite coordinates') from exc
        if len(target) != 3 or not all(math.isfinite(v) for v in target):
            raise ValueError('Target must contain three finite coordinates')
        now = time.monotonic() if now_s is None else float(now_s)
        self.target = target
        self.state = ROVState.APPROACH
        self.arrived = False
        self.acknowledged = False
        self.started_at = now
        self.history.append(f'Simulated approach requested: {target} m')
        try:
            self.adapter.request_approach(target, now)
        except Exception as exc:
            self._fault(f'Approach command failed: {exc}')

    def approve_collection(self, now_s=None):
        """Require operator approval after confirmed simulated arrival."""
        if self.state != ROVState.APPROACH or not self.arrived or not self.acknowledged:
            raise RuntimeError('Wait for acknowledged arrival before approving collection')
        now = time.monotonic() if now_s is None else float(now_s)
        self.state = ROVState.COLLECT
        self.acknowledged = False
        self.started_at = now
        self.history.append('Operator approved simulated collection')
        try:
            self.adapter.request_collection(self.target, now)
        except Exception as exc:
            self._fault(f'Collection command failed: {exc}')

    def tick(self, now_s=None):
        """Consume simulated feedback and enforce acknowledgement/completion deadlines."""
        if self.state not in (ROVState.APPROACH, ROVState.COLLECT):
            return
        now = time.monotonic() if now_s is None else float(now_s)
        if now < self.started_at:
            raise ValueError('Controller time moved backwards')
        if self.state == ROVState.APPROACH and self.arrived:
            return  # Hold for manual approval; arrival has already been confirmed.
        if not self.acknowledged and now - self.started_at > self.ack_timeout_s:
            self._fault('Command acknowledgement timed out')
            return
        if now - self.started_at > self.action_timeout_s:
            self._fault('Action completion timed out')
            return
        try:
            events = self.adapter.poll(now)
        except Exception as exc:
            self._fault(f'Feedback unavailable: {exc}')
            return
        for event in events:
            if self.state == ROVState.FAULT:
                break
            if event == 'rejected':
                self._fault('Simulated command rejected')
            elif self.state == ROVState.APPROACH:
                if event == 'approach_ack':
                    self.acknowledged = True
                    self.history.append('Approach acknowledged')
                elif event == 'arrived' and self.acknowledged:
                    self.arrived = True
                    self.history.append('Simulated arrival confirmed; awaiting operator approval')
            elif self.state == ROVState.COLLECT:
                if event == 'collection_ack':
                    self.acknowledged = True
                    self.history.append('Collection acknowledged')
                elif event == 'complete' and self.acknowledged:
                    self.history.append('Simulated collection complete')
                    self.state = ROVState.IDLE
                    self.armed = False
                    self.arrived = False
                    self.target = None
                    self.started_at = None

    def stop(self, reason='Operator stop'):
        """Latch FAULT and request a simulated stop; no hardware E-stop exists."""
        self._fault(reason)

    def reset(self):
        """An explicit operator reset clears the simulated fault latch."""
        if self.state != ROVState.FAULT:
            raise RuntimeError('Reset is only available after a fault/stop')
        self.state = ROVState.IDLE
        self.armed = False
        self.arrived = False
        self.acknowledged = False
        self.target = None
        self.started_at = None
        self.fault_reason = None
        self.history.append('Operator reset simulated controller')

    def _fault(self, reason):
        self.state = ROVState.FAULT
        self.fault_reason = str(reason)
        self.armed = False
        self.arrived = False
        self.acknowledged = False
        self.history.append(f'FAULT: {reason}')
        try:
            self.adapter.safe_stop()
        except Exception as exc:
            self.fault_reason += f'; stop request failed: {exc}'
            self.history.append(f'FAULT: stop request failed: {exc}')
