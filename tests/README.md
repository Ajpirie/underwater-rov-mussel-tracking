# Integration tests

From the repository root:

```bash
python -m unittest discover -s tests -v
```

The suite checks recorded data reconstruction, the thread-safe observation queue, individual-ID UKF updates, time-based expiry, causal forecast scoring, numerical timing equivalence and output exports. Actual predictor tests require **FilterPy** and explicitly skip if it is absent. Passing only the dependency-independent tests does not establish numerical UKF correctness.

For a GUI smoke test, use `python -m GUI` to replay the 15 fps dataset with a 1 s horizon, then test the live demo:

```bash
python -m Integration.example_camera_feed
```

Run mode should show sequential observations of two IDs without any call to `finish_frame()`. No ROV hardware commands are implemented.
