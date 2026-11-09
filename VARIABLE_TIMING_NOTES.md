# Variable frame timing — change record

The supported UKF now accepts `predictPositionUKF(elapsed_s=...)`. State transition and process noise use this elapsed interval, while the forecast still covers `timestep * num_timesteps` seconds. The damped-velocity equations, noise formulae, gating rules and default tuning values were not changed. Calling the method without `elapsed_s` retains nominal fixed-rate behaviour.

- `Integration/live_input.py`: timestamps every individual Run observation with a monotonic clock on receipt. No camera frame boundary is required. These are **software arrival times**, not exposure timestamps.
- `Integration/tracking_session.py`: passes actual recorded frame intervals to the UKF; measures ID expiry and centroid speed in seconds; forecasts are scored causally when their target time is reached. For off-grid target times, scoring interpolates between valid consecutive observations after the later one arrives, not before.
- `Integration/observation_session.py`: maintains per-ID elapsed timing for Run; only the observed mussel is updated. It scores against later measurements for that ID and expires tracks in seconds.
- `GUI/control_panel.py`: labels ID timeout in seconds and keeps Test (frame replay) distinct from Run (observation stream).
- `Integration/session_export.py`: records time-based timeout and adds a time-since-last-detection diagnostic.

**Limits:** No timestamps or frame boundary markers can be recovered from `(x, y, z, ID)` values alone. Run no longer requires frame-boundary signals. Variable upstream vision/transport delay biases the inferred frame time; future camera integration should expose exposure timestamps if available. No ROV actuation changes were made.

## Verification

Run `python -m unittest discover -s tests -v` in the repository. Repeat the external UKF equivalence check from the previous refactor with the new file to verify nominal-time predictions against the submitted baseline. Run tests with native FilterPy; a passing run with skipped UKF tests does not validate UKF numerical behaviour.

Submitted reports are unchanged. The retired UKF has been removed from the active repository tree.
