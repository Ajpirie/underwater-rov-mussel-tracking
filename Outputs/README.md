# Outputs — generated session recordings

The GUI creates files named like:

`test_RD_004_YYYYMMDD_HHMMSS_mmmmmm_observations.csv`
`test_RD_004_YYYYMMDD_HHMMSS_mmmmmm_predictions.csv`
`test_RD_004_YYYYMMDD_HHMMSS_mmmmmm_summary.json`

Run mode uses `run_camera_tracking_...` instead of a test dataset name. Filenames include mode, source and start-of-export timestamp to avoid overwrites. Test mode records the dataset frames (including empty frames); Run mode records each received observation separately without inventing frame boundaries.

- **Observations CSV:** Test has a `frame` column; Run has an `event_index` column. Both record arrival/dataset time (s), ID and X/Y/Z (m).
- **Predictions CSV:** raw and filtered positions (m), estimated velocity (m/s), time since last detection (s), per-ID observation interval or recorded frame interval (s), forecast target time (s), forecast position (m), rejected-measurement flag, future-target error and hold baseline error (m). A forecast error is blank until its later target observation has actually arrived.
- **Summary JSON:** method (`prediction_method`), configuration, processed frame count (Test) or observation count (Run), evaluation protocol, aggregate metrics and ROV integration status.

These files are runtime outputs and **do not** replace the read-only original datasets in `Data/`. They can be replayed by converting the observations into the required `xpos,ypos,zpos,ID` CSV format if necessary; the exported observations are not themselves identical to the input schema.
