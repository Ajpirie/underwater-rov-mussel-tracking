"""Tests for the integration layer. Requires FilterPy for predictor tests."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
import numpy as np
from pathlib import Path

from Integration.live_input import LiveObservationBuffer, Observation
from unittest.mock import patch
from Integration.tracking_session import (ROOT, Frame, TrackingSession,
                                          discover_datasets, load_csv_frames, recommended_dt)
from Integration.session_export import export_session
from Integration.observation_session import ObservationSession


class DataTests(unittest.TestCase):
    def test_repository_datasets(self):
        files = discover_datasets()
        self.assertEqual(len(files), 6)
        for path in files:
            frames = load_csv_frames(path, recommended_dt(path))
            self.assertTrue(frames)
            self.assertEqual([f.index for f in frames], list(range(len(frames))))

    def test_15fps_reconstructs_frames(self):
        frames = load_csv_frames(ROOT / 'Data/Real/15fps/260721.csv', 1 / 15)
        self.assertEqual((len(frames), sum(len(f.detections) for f in frames)), (60, 702))

    def test_missing_data_is_not_a_track(self):
        frames = load_csv_frames(ROOT / 'Data/Real/RD_004.csv', 0.5)
        self.assertEqual(len(frames), 248)
        self.assertTrue(any(not f.detections for f in frames))
        self.assertFalse(any(0 in f.detections for f in frames))

    def test_reject_bad_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.csv'
            path.write_text('xpos,ypos,zpos,ID\n0,invalid,1,2\n')
            with self.assertRaisesRegex(ValueError, 'line 2'):
                load_csv_frames(path, .5)


class BridgeTests(unittest.TestCase):
    def test_sequential_ids_need_no_frame_boundary(self):
        source = LiveObservationBuffer()
        source.submit_detection(5, .1, .2, .3)
        source.submit_detection(6, -.1, .4, .5)
        events = source.read()
        self.assertEqual([o.mussel_id for o in events], [5, 6])
        self.assertEqual(events[0].position, (.1, .2, .3))
        self.assertTrue(events[0].time_s <= events[1].time_s)

    def test_repeated_id_is_a_new_observation(self):
        source = LiveObservationBuffer()
        source.submit_detection(5, .1, .2, .3)
        source.submit_detection(5, .2, .2, .3)
        self.assertEqual(len(source.read()), 2)

    def test_invalid_observations_rejected(self):
        source = LiveObservationBuffer()
        for mid in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                source.submit_detection(mid, 1, 2, 3)
        with self.assertRaises(ValueError):
            source.submit_detection(1, float('nan'), 2, 3)

    def test_monotonic_arrival_timestamps(self):
        source = LiveObservationBuffer()
        with patch('Integration.live_input.time.monotonic_ns', side_effect=(100_000_000_000, 100_050_000_000)):
            source.clear()
            source.submit_detection(5, .1, .2, .3)
        self.assertAlmostEqual(source.read()[0].time_s, .05)

    def test_clear_resets_queue_and_clock(self):
        source = LiveObservationBuffer()
        source.submit_detection(5, .1, .2, .3)
        source.clear()
        self.assertFalse(source.read())
        self.assertEqual(source.dropped_observations, 0)

    def test_overflow_flag(self):
        source = LiveObservationBuffer(max_observations=1)
        source.submit_detection(1, .1, .2, .3)
        source.submit_detection(2, .1, .2, .3)
        self.assertEqual(source.dropped_observations, 1)
        self.assertEqual(source.read()[0].mussel_id, 2)


@unittest.skipUnless(__import__('importlib').util.find_spec('filterpy'),
                     'FilterPy is required for the UKF')
class PredictionTests(unittest.TestCase):
    def test_scoring_is_causal(self):
        s = TrackingSession(.5, 1.0)
        for i in range(3):
            snap = s.process(Frame(i, .5 * i, {3: (.2 + .002 * i, 0., .3)}))
            self.assertEqual(snap.n_scores, 0)
        self.assertEqual(s.process(Frame(3, 1.5, {3: (.206, 0., .3)})).n_scores, 1)

    def test_all_datasets_run(self):
        for path in discover_datasets():
            with self.subTest(path=path.name):
                dt = recommended_dt(path)
                s = TrackingSession(dt, dt * 10)
                for f in load_csv_frames(path, dt):
                    s.process(f)
                self.assertGreater(len(s.tracks), 0)

    def test_exports(self):
        s = TrackingSession(.5, 1)
        frames = [Frame(i, i * .5, {2: (.2 + i * .002, .0, .3)}) for i in range(6)]
        for f in frames:
            s.process(f)
        with tempfile.TemporaryDirectory() as tmp:
            files = export_session(s, frames, 'Test', 'RD_001.csv', Path(tmp), 'example')
            self.assertEqual(len(files), 3)
            with files[0].open(newline='') as f:
                self.assertEqual(len(list(csv.DictReader(f))), 6)
            with files[1].open(newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(rows)
            self.assertIn('velocity_x_mps', rows[0])
            metadata = json.loads(files[2].read_text())
            self.assertEqual(metadata['method'], 'prediction_method')
            
    def test_nominal_elapsed_matches_default(self):
        from Prediction import prediction_method as psi
        a = psi.predictionInstance((.2, .0, .3, 1), timestep=.5, num_timesteps=2)
        b = psi.predictionInstance((.2, .0, .3, 1), timestep=.5, num_timesteps=2)
        for i in range(8):
            if i:
                pos = (.2 + .001 * i, .0, .3)
                for p in (a, b):
                    p.x_history.append(pos[0]); p.y_history.append(pos[1]); p.z_history.append(pos[2])
            a.predictPositionUKF()
            b.predictPositionUKF(elapsed_s=.5)
            np.testing.assert_allclose(a.ukf.x, b.ukf.x, rtol=0, atol=0)
            np.testing.assert_allclose(a.ukf.P, b.ukf.P, rtol=0, atol=0)
            np.testing.assert_allclose(a.ukf.Q, b.ukf.Q, rtol=0, atol=0)
            np.testing.assert_allclose(a.x_future_history, b.x_future_history, rtol=0, atol=0)

    def test_irregular_timestep_changes_process_noise(self):
        from Prediction import prediction_method as psi
        p = psi.predictionInstance((.2, .0, .3, 1), timestep=.5, num_timesteps=2)
        p.predictPositionUKF()
        p.x_history.append(.201); p.y_history.append(.0); p.z_history.append(.3)
        p.predictPositionUKF(elapsed_s=.31)
        np.testing.assert_allclose(p.Q_base, psi._processNoise(.31, p.tuning), atol=0, rtol=0)
        self.assertNotAlmostEqual(float(p.Q_base[0, 0]),
                                  float(psi._processNoise(.5, p.tuning)[0, 0]))

    def test_irregular_deadline_is_causal(self):
        s = TrackingSession(.25, 1.0)
        times = [0.0, .25, .75, 1.4]
        for i, t in enumerate(times):
            snap = s.process(Frame(i, t, {3: (.2 + .01 * t, 0., .3)}))
            if i < 3:
                self.assertEqual(snap.n_scores, 0)
        self.assertEqual(snap.n_scores, 1)
        self.assertAlmostEqual(s.scores[0].target_time_s, 1.25)
        self.assertAlmostEqual(s.scores[0].time_s, 1.4)

    def test_id_expiry_is_time_based(self):
        s = TrackingSession(.5, 1.0, stale_after_s=.8)
        s.process(Frame(0, 0.0, {3: (.2, .0, .3)}))
        original = s.tracks[3].predictor
        s.process(Frame(1, .2, {}))
        s.process(Frame(2, .6, {3: (.201, .0, .3)}))
        self.assertIs(s.tracks[3].predictor, original)
        s.process(Frame(3, 1.5, {3: (.202, .0, .3)}))
        self.assertIsNot(s.tracks[3].predictor, original)

    def test_rejects_non_increasing_frame_time(self):
        s = TrackingSession(.5, 1.0)
        s.process(Frame(0, 0, {3: (.2, .0, .3)}))
        with self.assertRaisesRegex(ValueError, 'increase'):
            s.process(Frame(1, 0, {3: (.2, .0, .3)}))

    def test_independent_sessions(self):
        import numpy as np

        def frames(dt):
            return [
                Frame(i, i * dt, {1: (0.1 + i * 0.001, 0.02, 0.3)})
                for i in range(8)
            ]

        def run(session, data):
            return [
                session.process(f).tracks[1]["forecast"]
                for f in data
            ]

        dt_a = 0.5
        dt_b = 1 / 15

        # Independent reference runs
        expected_a = run(TrackingSession(dt_a, 1.0), frames(dt_a))
        expected_b = run(TrackingSession(dt_b, 1.0), frames(dt_b))

        # Two sessions operating together
        a = TrackingSession(dt_a, 1.0)
        b = TrackingSession(dt_b, 1.0)

        actual_a, actual_b = [], []

        for fa, fb in zip(frames(dt_a), frames(dt_b)):
            actual_a.append(a.process(fa).tracks[1]["forecast"])
            actual_b.append(b.process(fb).tracks[1]["forecast"])

        np.testing.assert_allclose(actual_a, expected_a, atol=1e-12, rtol=0)
        np.testing.assert_allclose(actual_b, expected_b, atol=1e-12, rtol=0)


@unittest.skipUnless(__import__('importlib').util.find_spec('filterpy'),
                     'FilterPy required for observation-stream prediction tests')
class ObservationTests(unittest.TestCase):
    def test_interleaving_updates_only_matching_id(self):
        session = ObservationSession(.1, .5, stale_after_s=2)
        session.observe(Observation(1, (.1, 0, .3), .1))
        original = session.tracks[1].predictor
        self.assertEqual(len(original.x_future_history), 1)
        session.observe(Observation(2, (.2, 0, .4), .2))
        self.assertEqual(len(original.x_future_history), 1)
        session.observe(Observation(1, (.11, 0, .3), .34))
        self.assertIs(session.tracks[1].predictor, original)
        self.assertEqual(len(original.x_future_history), 2)
        self.assertAlmostEqual(original._process_dt, .24)

    def test_time_based_expiry_after_silence(self):
        session = ObservationSession(.1, .5, stale_after_s=.5)
        session.observe(Observation(1, (.1, 0, .3), .1))
        old = session.tracks[1].predictor
        self.assertIsNone(session.expire_at(.4))
        cleared = session.expire_at(.61)
        self.assertIsNotNone(cleared)
        self.assertFalse(cleared.tracks)
        session.observe(Observation(1, (.2, 0, .3), 1.0))
        self.assertIsNot(session.tracks[1].predictor, old)

    def test_scoring_waits_for_actual_future_observation(self):
        session = ObservationSession(.1, .5, stale_after_s=2)
        for i, time_s in enumerate((.1, .3, .6, .9)):
            snap = session.observe(Observation(1, (.1 + .01 * i, 0, .3), time_s))
            if i < 3:
                self.assertEqual(snap.n_scores, 0)
        self.assertEqual(snap.n_scores, 1)
        self.assertAlmostEqual(session.scores[0].target_time_s, .8)
        self.assertAlmostEqual(session.scores[0].time_s, .9)

    def test_run_export_observations_not_frames(self):
        session = ObservationSession(.1, .5, stale_after_s=2)
        events = [Observation(1, (.1, 0, .3), .1),
                  Observation(2, (.2, 0, .4), .2),
                  Observation(1, (.11, 0, .3), .3)]
        for event in events:
            session.observe(event)
        with tempfile.TemporaryDirectory() as d:
            paths = export_session(session, events, 'Run', 'camera_tracking', Path(d), 'test')
            with paths[0].open(newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), len(events))
            self.assertEqual(rows[0]['mussel_id'], '1')
            self.assertIn('event_index', rows[0])
            with paths[1].open(newline='') as f:
                self.assertIn('velocity_x_mps', next(csv.DictReader(f)))
            meta = json.loads(paths[2].read_text())
            self.assertEqual(meta['observation_count'], 3)
            self.assertIn('no camera frame boundaries', meta['input_timing'])

    def test_export_event_indices_after_idle_expiry(self):
        session = ObservationSession(.1, .2, stale_after_s=.25)
        events = [Observation(1, (.1, 0, .3), .1), Observation(1, (.11, 0, .3), .6)]
        session.observe(events[0])
        session.expire_at(.4)  # Display-only snapshot, not an observation.
        session.observe(events[1])
        with tempfile.TemporaryDirectory() as d:
            paths = export_session(session, events, 'Run', 'camera_tracking', Path(d), 'test')
            with paths[0].open(newline='') as f:
                raw = list(csv.DictReader(f))
            with paths[1].open(newline='') as f:
                predicted = list(csv.DictReader(f))
            self.assertEqual([r['event_index'] for r in raw], ['0', '1'])
            self.assertEqual([r['event_index'] for r in predicted], ['0', '1'])

    def test_repeated_timestamps_same_id_are_rejected(self):
        session = ObservationSession(.1, .5)
        session.observe(Observation(1, (.1, 0, .3), .1))
        with self.assertRaisesRegex(ValueError, 'same timestamp'):
            session.observe(Observation(1, (.2, 0, .3), .1))
