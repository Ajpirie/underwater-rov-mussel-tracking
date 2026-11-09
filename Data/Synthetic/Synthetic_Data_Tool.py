"""
Builds synthetic multi-mussel tracking CSV data by re-using REAL recorded
trajectory segments from RD_001-RD_004, re-labelling each as a distinct
mussel ID, and interleaving them frame-by-frame to simulate several mussels
being tracked simultaneously by the same camera/rig.

No position values are invented. Every (x, y, z) triple in the output was
actually recorded in one of the source files. The only synthetic elements
are:
  1. Re-assigned mussel IDs (so trajectories from different source files
     don't collide on ID).
  2. A small constant spatial offset per mussel (a few cm), because these
     source recordings were each captured as a SINGLE mussel in the same
     tank/coordinate frame -- overlaying their raw coordinates would make
     multiple "mussels" occupy the same point in space, which never
     happens in reality. The offset separates them the way mussels
     attached at different points on a rope/tray would be, while leaving
     each trajectory's own motion (noise, drift, oscillation, dropout
     frames) completely untouched.
  3. Staggered start frames, so mussels enter/leave the scene at
     different times -- exactly like real tracking footage, and a good
     stress test for "pause/stop prediction when ID lost" logic.

Dropout convention: rows where a mussel was not detected are written as
(0, 0, 0) under that SAME mussel ID (matching the convention already used
in RD_001/RD_002/RD_003), not offset, so predictPositionUKF's
`np.all(current_meas == 0)` dropout check still works correctly.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

SOURCE_DIR = "Data/Real/"
OUT_PATH = "Data/Synthetic/SD_001.csv"


def load_segment(filename, id_filter):
    df = pd.read_csv(SOURCE_DIR + filename)
    seg = df[df["ID"] == id_filter][["xpos", "ypos", "zpos"]].to_numpy(dtype=float)
    return seg


def apply_offset(segment, dx, dy, dz):
    """Shift all NON-dropout rows by a constant offset. Dropout rows
    (exact 0,0,0 -- the 'not detected this frame' sentinel) are left
    untouched so the dropout check in predictPositionUKF still fires."""
    out = segment.copy()
    is_dropout = np.all(out == 0.0, axis=1)
    out[~is_dropout, 0] += dx
    out[~is_dropout, 1] += dy
    out[~is_dropout, 2] += dz
    return out


# ---------------------------------------------------------------------------
# 1. Pull real trajectory segments (each is a genuinely-recorded mussel track)
# ---------------------------------------------------------------------------
seg_1 = load_segment("RD_001.csv", 1.0)  # 98 rows  -> new mussel ID 1
seg_2 = load_segment("RD_003.csv", 15)  # 49 rows  -> new mussel ID 2
seg_3_df = pd.read_csv(SOURCE_DIR + "RD_004.csv")  # keep ALL 248 rows (incl.
seg_3 = seg_3_df[["xpos", "ypos", "zpos"]].to_numpy(dtype=float)  # its own dropout frames) -> new mussel ID 3
seg_4 = load_segment("RD_003.csv", 1)  # 66 rows  -> new mussel ID 4

print("Segment lengths:", len(seg_1), len(seg_2), len(seg_3), len(seg_4))

# ---------------------------------------------------------------------------
# 2. Spatially separate the mussels (small constant offsets, dropout-safe)
# ---------------------------------------------------------------------------
seg_1 = apply_offset(seg_1, 0.00, 0.00, 0.00)  # baseline / reference position
seg_2 = apply_offset(seg_2, 0.08, 0.04, 0.02)
seg_3 = apply_offset(seg_3, -0.06, -0.03, -0.01)
seg_4 = apply_offset(seg_4, 0.04, -0.05, 0.03)

# ---------------------------------------------------------------------------
# 3. Stagger entry/exit so mussels overlap realistically (incl. a 4-way
#    overlap window to stress-test simultaneous multi-ID tracking)
# ---------------------------------------------------------------------------
mussels = [
    {"id": 1, "data": seg_1, "start": 0},
    {"id": 2, "data": seg_2, "start": 60},
    {"id": 3, "data": seg_3, "start": 0},
    {"id": 4, "data": seg_4, "start": 70},
]

for m in mussels:
    m["end"] = m["start"] + len(m["data"]) - 1  # inclusive last frame

total_frames = max(m["end"] for m in mussels) + 1

print("\nMussel windows (inclusive frame indices):")
for m in mussels:
    print(f"  ID {m['id']}: frames {m['start']}-{m['end']}  ({len(m['data'])} rows)")
print(f"Total timeline length: {total_frames} frames")

# ---------------------------------------------------------------------------
# 4. Interleave: for each frame, emit one row per currently-active mussel,
#    in ascending ID order (a stand-in for detection order within a frame)
# ---------------------------------------------------------------------------
rows = []
concurrency = np.zeros(total_frames, dtype=int)

for frame in range(total_frames):
    active_count = 0
    for m in mussels:
        if m["start"] <= frame <= m["end"]:
            local_idx = frame - m["start"]
            x, y, z = m["data"][local_idx]
            rows.append((x, y, z, m["id"]))
            active_count += 1
    concurrency[frame] = active_count

out_df = pd.DataFrame(rows, columns=["xpos", "ypos", "zpos", "ID"])

# ---------------------------------------------------------------------------
# 5. Report + write
# ---------------------------------------------------------------------------
print(f"\nTotal output rows: {len(out_df)}")
print("Concurrency histogram (how many frames have N mussels active):")
for n in range(1, 5):
    print(f"  {n} concurrent: {(concurrency == n).sum()} frames")

out_df.to_csv(OUT_PATH, index=False, float_format="%.8f")
print(f"\nWrote {OUT_PATH}")
