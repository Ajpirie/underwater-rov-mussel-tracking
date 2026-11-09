"""Export replay and live-session observations, estimates and evaluation."""
from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import re

from Integration.tracking_session import ROOT, Frame, TrackingSession
from Integration.live_input import Observation

OUTPUT_DIR = ROOT / 'Outputs'


def export_session(session: TrackingSession, frames: list[Frame | Observation], mode: str, source: str,
                   directory: Path = OUTPUT_DIR, run_label: str | None = None) -> list[Path]:
    """Save inputs, per-track predictions and summary without overwriting previous runs."""
    directory.mkdir(parents=True, exist_ok=True)
    source_name = re.sub(r'[^A-Za-z0-9_-]+', '_', Path(source).stem or 'camera').strip('_')[:38]
    stamp = run_label or datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    prefix = f'{mode.lower()}_{source_name}_{stamp}'
    raw_path = directory / f'{prefix}_observations.csv'
    prediction_path = directory / f'{prefix}_predictions.csv'
    meta_path = directory / f'{prefix}_summary.json'

    is_stream = mode.lower() == 'run'
    with raw_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['event_index' if is_stream else 'frame', 'time_s',
                         'mussel_id', 'x_m', 'y_m', 'z_m'])
        for index, item in enumerate(frames):
            if isinstance(item, Observation):
                writer.writerow([index, item.time_s, item.mussel_id, *item.position])
            else:
                if not item.detections:
                    writer.writerow([item.index, item.time_s, '', '', '', ''])
                for mid, position in sorted(item.detections.items()):
                    writer.writerow([item.index, item.time_s, mid, *position])

    # Error belongs to the target frame, not the frame that issued the forecast.
    scores = {(score.origin_frame, score.mussel_id): score for score in session.scores}
    with prediction_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['event_index' if is_stream else 'frame', 'time_s', 'mussel_id', 'observed',
                         'raw_x_m', 'raw_y_m', 'raw_z_m',
                         'filtered_x_m', 'filtered_y_m', 'filtered_z_m',
                         'velocity_x_mps', 'velocity_y_mps', 'velocity_z_mps',
                         'forecast_for_time_s', 'forecast_x_m', 'forecast_y_m', 'forecast_z_m',
                         'measurement_rejected', 'forecast_error_m', 'hold_error_m',
                         'seconds_since_seen', 'frame_interval_s'])
        for mid, track in sorted(session.tracks.items()):
            for row in track.history:
                score = scores.get((row['frame'], mid))
                raw = row['raw'] if row['raw'] is not None else ('', '', '')
                writer.writerow([row.get('observation_index', row['frame']) if is_stream else row['frame'],
                                 row['time_s'], mid, int(row['fresh']),
                                 *raw, *row['filtered'], *row['velocity'],
                                 row['time_s'] + session.horizon_s, *row['forecast'],
                                 int(row['rejected']), score.error_m if score else '',
                                 score.hold_m if score else '', row.get('age_s', 0.0),
                                 row.get('frame_interval_s', session.dt)])

    meta = {
        'mode': mode, 'source': source, 'method': session.model,
        'nominal_frame_period_s': session.dt, 'prediction_horizon_s': session.horizon_s,
        'id_timeout_s': session.stale_after_s,
        'observation_count' if is_stream else 'frame_count': len(frames),
        'input_timing': ('timestamped at adapter receipt; no camera frame boundaries or capture timestamps'
                         if is_stream else 'recorded dataset frame times'),
        'last_processed_snapshot' if is_stream else 'last_processed_frame': len(session.snapshots) - 1,
        'scoring': '3D error at forecast target time; interpolate between two valid measurements only after the later observation arrives',
        'metrics': session.metrics_at(len(session.snapshots) - 1) if session.snapshots else {},
        'units': 'positions: m; velocities: m/s; times: s; errors: m',
        'rov': 'advisory only; no movement/collection hardware integration',
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return [raw_path, prediction_path, meta_path]
