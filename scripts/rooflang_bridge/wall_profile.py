"""Opt-in, exclusive wall-clock accounting for one worker RPC.

These timings never enter simulated costs. Parent scopes exclude nested scopes;
response pipe writes and the small telemetry suffix belong to the caller's RPC
residual, not the response serialization measurement.
"""

import json
import re
from contextlib import contextmanager, nullcontext
from time import perf_counter

_PROFILE_FLAG = re.compile(r'"profile_wall"\s*:\s*true\b')


class WallProfile:
    def __init__(self):
        self.sections = {}
        self._stack = []
        self._last = None

    def _accrue(self, now):
        if self._stack:
            self.sections[self._stack[-1]]["seconds"] += now - self._last
        self._last = now

    @contextmanager
    def scope(self, name):
        self._accrue(perf_counter())
        row = self.sections.setdefault(name, {"calls": 0, "seconds": 0.0})
        row["calls"] += 1
        self._stack.append(name)
        try:
            yield
        finally:
            self._accrue(perf_counter())
            self._stack.pop()

    def call(self, name, function, *args, **kwargs):
        # Pricing executes tens of thousands of times per round. Avoid creating
        # a context manager for each call while retaining exclusive accounting.
        self._accrue(perf_counter())
        row = self.sections.setdefault(name, {"calls": 0, "seconds": 0.0})
        row["calls"] += 1
        self._stack.append(name)
        try:
            return function(*args, **kwargs)
        finally:
            now = perf_counter()
            row["seconds"] += now - self._last
            self._last = now
            self._stack.pop()


def scope(profile, name):
    return profile.scope(name) if profile is not None else nullcontext()


def request_profile(line):
    # Detect the opt-in before json.loads so its time is observable without
    # reading a clock for normal requests. The decoded top-level flag remains
    # authoritative; nested/string occurrences do not enable exported timings.
    if '"profile_wall"' in line and _PROFILE_FLAG.search(line):
        return WallProfile()
    return None


def encode_reply(reply, profile):
    with scope(profile, "response_encode"):
        payload = json.dumps(reply, allow_nan=False)
    if profile is None:
        return payload
    # Serialize the potentially large graph only once. The telemetry has no
    # simulated meaning and is attached after measuring the ordinary response.
    suffix = json.dumps({"sections": profile.sections}, allow_nan=False)
    return payload[:-1] + ', "wall_profile": ' + suffix + "}"
