# Handover change record

**Source:** The user-supplied `underwater-rov-mussel-tracking.zip` (9 November 2026). This update does not modify any report, camera dataset, UKF calculation or tuned parameter. The active module was renamed, and only the obsolete method-selection and comparison paths were removed.

## Supported prediction module

- The formerly suffixed active predictor was **renamed** to `Prediction/prediction_method.py`. Its module documentation was shortened; the UKF algorithms and default tuning were preserved.
- `Prediction/__init__.py` now points its Python 3.9 import-compatibility loader at `prediction_method.py`.
- `Integration/tracking_session.py` imports this method directly. There is no legacy switch or alternate UKF constructor. Run and Test still use the same prediction model.
- The benchmark was renamed to `Test_Verification/Benchmark_UKF.py`, benchmarking the active UKF and its ablations against zero-order hold. It no longer runs the retired UKF.
- `Test_Verification/lull_detector.py` retains the historical LullDetector calculation as a separate analysis helper. The two active lull scripts now import that helper.
- The retired predictor implementation, unsupported archived GUIs/runtime, and archived verification scripts requiring that implementation were removed. Unrelated Fourier and research experiments remain.

## Documentation

A compact diagram is available at `Images/System_Architecture.svg`, with editable Mermaid source at `Images/System_Architecture.md`. The root, Prediction, Integration and Test/Verification READMEs describe the supported paths only.

## Validation and scope

Run `python -m unittest discover -s tests -v` with FilterPy installed, then `python -m GUI`. Recheck any separate numerical-equivalence baseline after changing the UKF mathematics or tuning; **this rename alone is not a model change**. The input adapter is still in-process and receipt-timestamped; ROV control remains placeholders and is not safe for hardware actuation.

**Unchanged:** All submitted documents in `Docs/` and all data in `Data/`. ROV actuator code, camera coordinate calibration and the model's actual equations are not changed in this delivery.
