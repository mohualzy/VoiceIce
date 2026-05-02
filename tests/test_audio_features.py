import unittest
from unittest.mock import patch

import numpy as np
import plotly.graph_objects as go

import utils


class AudioFeatureTests(unittest.TestCase):
    def test_pitch_by_temp_uses_supplied_sample_rate(self):
        audio = np.zeros(2048, dtype=float)

        with patch("librosa.effects.pitch_shift", return_value=audio) as pitch_shift:
            utils.process_audio_pitch_by_temp(audio, 1.2, sr=44100)

        self.assertEqual(pitch_shift.call_args.kwargs["sr"], 44100)

    def test_analyze_emotion_features_returns_real_acoustic_metrics(self):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        intense_audio = 0.5 * np.sin(2 * np.pi * 1000 * t)
        quiet_audio = 0.005 * np.sin(2 * np.pi * 220 * t)

        intense = utils.analyze_emotion_features(intense_audio, sr)
        quiet = utils.analyze_emotion_features(quiet_audio, sr)

        self.assertGreater(intense["rms"], 0.05)
        self.assertGreater(intense["zcr"], 0.1)
        self.assertEqual(intense["emotion"], "激烈")
        self.assertEqual(len(intense["mfcc_mean"]), 13)
        self.assertEqual(quiet["emotion"], "平静")

    def test_draw_f0_curve_returns_voiced_frequency_trace(self):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = 0.3 * np.sin(2 * np.pi * 220 * t)

        fig = utils.draw_f0_curve(audio, sr, "F0", "#87CEFA")

        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)
        f0_values = np.asarray(fig.data[0].y, dtype=float)
        self.assertGreater(np.count_nonzero(np.isfinite(f0_values)), 0)
        self.assertAlmostEqual(float(np.nanmedian(f0_values)), 220, delta=20)

    def test_draw_emotion_timeline_returns_normalized_peak_time(self):
        sr = 16000
        t = np.linspace(0, 0.5, sr // 2, endpoint=False)
        quiet = 0.01 * np.sin(2 * np.pi * 120 * t)
        intense = 0.5 * np.sin(2 * np.pi * 1200 * t)
        audio = np.concatenate([quiet, intense])

        fig = utils.draw_emotion_timeline(audio, sr)

        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)
        self.assertGreaterEqual(fig.layout.meta["peak_time"], 0.5)
        self.assertLessEqual(fig.layout.meta["peak_intensity"], 1.0)
        self.assertGreater(fig.layout.meta["peak_intensity"], 0.5)


if __name__ == "__main__":
    unittest.main()
