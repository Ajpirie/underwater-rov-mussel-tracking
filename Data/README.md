# Data — original recordings and synthetic tracking sets

**No recordings or data-generation source files have been changed.**

- `Real/RD_001.csv`–`RD_004.csv`: historical real tracking positions, generally one row per camera frame. The original recorded frame period used by the GUI is **0.5 s (2 fps)**; confirm this if another acquisition setup is used. `RD_004.csv` includes all-zero ID-0 dropout rows that are not real mussels.
- `Real/15fps/260721.csv`: multiple IDs detected in the same frame; `260721.txt` contains the companion tracker structure needed to reconstruct true frame boundaries. The GUI loader requires both files, and uses **1/15 s** frame period.
- `Synthetic/SD_001.csv`: synthetic data; repeated IDs mark the next frame under automatic grouping. `Synthetic_Data_Tool.py` is the unchanged generator.
- `Real/tracking.mp4` and supporting real-data logs/documents: retained for provenance.
- `Test/`: additional test/wave datasets, not necessarily mussel-tracking CSVs.

The GUI's CSV loader requires headers **`xpos,ypos,zpos,ID`**; values must be finite and positions in **metres**. Other datasets with one row per frame can use **Rows** grouping. **Repeated IDs** grouping starts a new frame when an ID appears again; this is a convention, not a universal reconstruction rule.

Camera coordinates are measured relative to the stereo camera lens centre (X left/right, Y up/down, Z forward/back). Axis polarities and any camera-to-vehicle transform still need confirmation before ROV actuation.
