# GUI — supervisory control panel

Preview: [Accuracy view](Preview_Accuracy.png) and [3D tracking view](Preview_3D.png).

**Launch:** from the repository root, run `python -m GUI`. Main entry: `GUI/control_panel.py` (`ControlPanel`, `launch`). The GUI is a display/operator interface; it does not implement a UKF or command the actual ROV.

## What you see

- **Test / Run:** Both modes share the same predictions, cluster status, charts and logging. Test uses a CSV and frame-by-frame playback; Run receives individual timestamped-on-arrival observations from `Integration.live_input`; no frame boundaries are needed.
- **Scene 3D:** Filtered mussel positions, forecast positions, current and forecast centroids, and camera-relative alignment target. Drag to rotate; Matplotlib toolbar offers pan/zoom/reset/save.
- **Position over time:** X, Y and Z in metres. Select centroid or an individual ID; toggle measured (dots), filtered (solid) and future forecast (dashed). The forecast is drawn at *future target time*, not creation time.
- **Accuracy:** Future-target 3D Euclidean error (mm) vs zero-order hold; per-axis RMSE bars (mm). Scores appear only when valid future measurements arrive. The supplied `260721.csv` 15 fps recording is about 3.93 s long, so the usual 5 s horizon cannot produce accuracy scores. For this short dataset, set the forecast horizon to 1 s or another value shorter than the recording before starting. The GUI logs a warning when no score can be produced.
- **Motion:** Filtered per-mussel speed in mm/s or rolling-path centroid speed and lull threshold; rejected measurements are flagged, and cumulative counts are shown.
- **Events:** Diagnostic messages, recording activity and outlier rejections.

## Controls

**Start / Pause / Resume / Step / Stop** are across the top. Replay speed and timeline are Test-only. **Lock current cluster** freezes the target IDs for advisory alignment so newly acquired IDs do not shift the selected group. **Apply alignment settings** updates the live advisory thresholds. A locked cluster is required before the interface presents a readiness assessment. **Request alignment/collection (stub)** calls placeholder functions and *never sends hardware commands*. Collection is disabled in Run mode because asynchronous tracking cannot establish collection readiness; neither stub is a safety interlock.

**Prediction settings:** nominal frame period (s), forecast horizon (s), ID timeout (s) and max ID. Run mode measures each ID’s observation-to-observation interval automatically; Test mode uses the recording's time base. **Advisory settings:** target Z (m), XY and Z tolerances (m), and low-motion threshold (m/s), rolling window (s) and minimum calm duration (s). Thresholds are starting values, not validated limits for deployment. The sole supported UKF method is fixed in the GUI by design.

Settings persist across launches in `~/.mussel_prediction/control_panel.json`; old saved ID-retention frame counts are converted to seconds once, using the last saved nominal period. Saved settings are operator preferences, not the UKF's internally validated tuning. Data exports are generated in repository `Outputs/` on completion/Stop or via **Save session**. See `Outputs/README.md`.

## Code

| File | What it does |
|---|---|
| `control_panel.py` | Builds Tkinter widgets and Matplotlib plots; manages replay, settings, progress, live polling, chart selection and placeholder buttons. |
| `__main__.py` | Supports `python -m GUI`. |

The active GUI imports the headless integration API instead of duplicating prediction logic. It uses Tkinter's main thread for drawing; the external CV adapter can publish observations from its own worker thread. Display refresh and plotting can be slower than the camera frame rate on low-power laptops; **a buffer overflow stops the session rather than silently losing measurements**.

**Architecture:** [System module diagram](../Images/System_Architecture.svg). The graphical interface calls `Integration/` rather than importing or selecting prediction variants.
