# Integration — tracking, session analysis and ROV simulation

This layer adapts external tracking records to `Prediction/prediction_method.py`, maintains per-ID state, evaluates forecasts and provides the supervisor with diagnostics. Neither the external CV transport nor ROV actuation is implemented. The new ROV sequence runs only against a simulated feedback adapter.

## Data flow

```mermaid
flowchart TB
    A["CV stream: ID, x, y, z"] --> B["live_input.py: timestamp + queue"]
    B --> C["ObservationSession: update received ID only"]
    D["Recorded CSV + frame map"] --> E["TrackingSession: step recorded frames"]
    C --> F["Prediction/prediction_method.py"]
    E --> F
    C --> G["Latest states, metrics, centroid and lulls"]
    E --> G
    G --> H["session_export.py → Outputs/"]
    G --> I["GUI supervision"]
    I -. "operator actions" .-> J["ROVController (simulation)"]
    J --> K["SimulatedROV feedback"]
```

**Run** uses independent observation-to-observation intervals measured on receipt; **Test** uses the recorded frame time. Both manage a separate UKF for each ID and forecast the same configured duration. [Simplified system diagram](../Images/System_Architecture.svg).

| File | Responsibility |
|---|---|
| `live_input.py` | Thread-safe queue of individual `(ID, x, y, z)` observations, timestamped at receipt. Public entry: `publish_observation()`. |
| `observation_session.py` | Run-mode per-ID updates, time-based expiry, causal forecast scoring and asynchronous cluster summaries. |
| `tracking_session.py` | Test-mode `Frame` replay, CSV loaders, per-frame UKF processing, historical tracking and scoring. Shared `Snapshot`/`Score` data types. |
| `session_export.py` | Observations, UKF histories and score/summary exports to `Outputs/`. |
| `example_camera_feed.py` | Optional in-process test stream for the GUI, not a CV implementation. |
| `rov_state_machine.py` | Four-state supervisory sequence, arming, approval, feedback handling, timeouts and fault latch. |
| `rov_simulator.py` | Fake command acknowledgements, arrival, collection completion and stop for offline tests. |
| `rov_interface.py` | Original ROV command placeholders; intentionally **not connected** to the simulator or GUI. |

## Live observation contract

The only assumed CV payload is `(mussel_id, x, y, z)`: a non-negative integer ID and three finite camera-relative positions **in metres**. IDs can arrive in any order and may repeat. No timestamp, camera frame delimiter or fixed delivery rate is required.

```python
from Integration.live_input import publish_observation

# Called as each new measurement becomes available (push or pull):
publish_observation(5, -0.10, 0.02, 0.42)
publish_observation(6, -0.06, 0.03, 0.44)
publish_observation(5, -0.101, 0.021, 0.420)
```

`publish_observation` is thread-safe and timestamps **at adapter receipt** using a monotonic clock. Call it after the GUI has entered Run mode; Start clears the previous queue and resets the clock origin. The GUI drains the queue and calls `ObservationSession.observe()` for every item. If the queue overflows, Run stops rather than silently producing an inaccurate timeline.

The external source can **push** records into a callback or be **polled** by an integration worker:

```python
# Example sketch; read_available is supplied by the eventual CV adapter.
for mussel_id, x, y, z in read_available():
    publish_observation(mussel_id, x, y, z)
```

There is no `finish_frame()` or batching requirement. Since no transport/API has been defined, this repository deliberately does not open a serial port, socket or camera device. If the CV system lives in a different process or machine, an adapter must receive its records and call the in-process function. The adapter must not run Tkinter calls outside the GUI thread.

## Live processing

`ObservationSession` maintains **one UKF per tracked ID**. When ID 5 arrives, only ID 5 is updated, with its own elapsed interval. ID 6 is not propagated or marked as a dropout merely because it was absent from ID 5's observation. A track expires once the time since its last observation exceeds `stale_after_s`. A later measurement of that ID starts a new filter; pending forecast scores for its old identity are discarded.

A `Snapshot` is created for each accepted input event and when an active track expires during silence. Its cluster centroid uses the last filtered positions of non-expired IDs; these positions are **not simultaneous**. `is_calm` and alignment readings are observational diagnostics only. With no frame grouping or capture timestamps, the exact frame-time centroid and wave-driven lull cannot be established. A separate, opt-in **simulated** approach and collection sequence is available in Run mode; live alignment/lull readings do not automatically trigger it. No vehicle commands are sent.

Prediction scores are formed only when a later valid measurement of the **same ID** reaches the forecast target time. If two observations bracket that target time, its position is interpolated after the second is received. No scores cross an ID-expiry gap. Arrival-time jitter and CV processing delay affect the timing accuracy.

## Recorded replay

`TrackingSession.process(Frame(index, time_s, detections))` is unchanged for Test mode. It reconstructs frames from the supplied CSVs, including the companion 15 fps tracker dump, and advances filters on each recorded timestep. The frame-based replay protocol remains available for reproducing recorded results. Its missing-frame/dropout handling is **not** applied to asynchronous Run observations.

## Simulation-only ROV controller

`ROVController` is a minimal state machine for testing the **supervisory sequence**, not the vehicle. It does not import the UKF, modify tracking state or initiate actions from lull/alignment heuristics.

```mermaid
flowchart LR
    I["IDLE"] -->|Arm + operator approach| A["APPROACH"]
    A -->|Arrival acknowledged + manual approval| C["COLLECT"]
    C -->|Completion feedback| I
    I -->|Operator stop| F["FAULT"]
    A -->|Stop / rejection / timeout| F
    C -->|Stop / rejection / timeout| F
    F -->|Explicit reset| I
```

- **IDLE:** operator arms the simulated controller and selects an active camera-relative target. Pressing **Simulate approach** freezes that target for this demo sequence.
- **APPROACH:** the simulator acknowledges the request and reports arrival after a configurable delay. The controller does not automatically advance after arrival; the operator must choose **Approve simulated collection**.
- **COLLECT:** the simulator acknowledges collection and reports completion; the controller disarms and returns to IDLE.
- **FAULT:** a rejected command, missing acknowledgement/completion, simulated stop, signal loss while active, or stopping tracking while armed/active causes a latched fault and a simulated stop request. **Reset simulated fault** explicitly returns to IDLE in a disarmed condition. Resetting never resumes a command.

The GUI checks the sequence during Run mode only. Test playback is never allowed to actuate or advance the supervisor. The demo adapter (`SimulatedROV`) emits feedback without connecting to hardware; a real adapter will require independently validated motor/tool status, coordinate calibration, controller-side interlocks, communication-loss handling and a physical E-stop. **`safe_stop()` here only clears simulated events. It cannot stop a real ROV.**

A small programmatic example (no CV input or GUI needed):

```python
from Integration.rov_state_machine import ROVController
from Integration.rov_simulator import SimulatedROV

control = ROVController(SimulatedROV())
control.arm()
control.start_approach((0.1, 0.0, 0.4))  # Camera-relative metres; demonstration only
# Repeatedly call control.tick() until control.arrived is True.
# Then call control.approve_collection() after explicit operator confirmation.
```

The controller has an acknowledgement deadline (default **1 s**) and action-completion deadline (default **15 s**) measured by a monotonic clock. These values are **simulation defaults**, not ROV operating or safety limits. They must be reassessed with real hardware.

## Handover limits

All reported spatial quantities use the camera lens-centre reference frame, not ROV body/world coordinates. Real ROV motor control, collection hardware, coordinate transforms, independently verified hardware interlocks and a physical emergency stop remain outside this code. The simulated operator approval and stop are not a substitute. Consult `VARIABLE_TIMING_NOTES.md` and the final report section references in the root README.
