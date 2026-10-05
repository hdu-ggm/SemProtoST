import unittest

import numpy as np
import pandas as pd

from baselines.NearestSensor.evaluate import (
    build_neighbor_map,
    haversine_distances_km,
    masked_metrics,
    transfer_predictions,
)


def _metadata():
    return pd.DataFrame(
        {
            "ID": [10, 11, 12, 13],
            "Lat": [0.0, 0.0, 0.0, 1.0],
            "Lng": [0.0, 1.0, 0.2, 0.0],
            "Fwy": ["I5", "I8", "I5", "I9"],
            "Direction": ["N", "N", "N", "S"],
        }
    )


class NearestSensorBaselineTest(unittest.TestCase):

    def test_haversine_distance_is_in_kilometers(self):
        distance = haversine_distances_km(
            0.0, 0.0, np.array([0.0]), np.array([1.0])
        )
        self.assertTrue(np.isclose(distance[0], 111.195, atol=0.01))

    def test_nearest_and_knn_transfer_use_only_seen_predictions(self):
        mapping = build_neighbor_map(_metadata(), known_indices=[0, 1], k=2)
        seen_prediction = np.array([[[[10.0], [30.0]]]])

        nearest = transfer_predictions(seen_prediction, mapping, "nearest")
        knn = transfer_predictions(seen_prediction, mapping, "knn")

        self.assertEqual(nearest.shape, (1, 1, 2, 1))
        self.assertEqual(nearest[0, 0, 0, 0], 10.0)
        expected = np.sum(np.array([10.0, 30.0]) * mapping.weights[0])
        self.assertTrue(np.isclose(knn[0, 0, 0, 0], expected))

    def test_road_filter_and_global_fallback_are_recorded(self):
        mapping = build_neighbor_map(
            _metadata(),
            known_indices=[0, 1],
            k=1,
            neighbor_filter="freeway_direction",
            fallback="global",
        )
        # Node 2 matches seen node 0; node 3 has no matching road/direction.
        self.assertEqual(mapping.neighbor_indices[0, 0], 0)
        self.assertFalse(mapping.used_fallback[0])
        self.assertTrue(mapping.used_fallback[1])

    def test_masked_metrics_match_expected_values(self):
        prediction = np.array([1.0, 4.0, 100.0])
        target = np.array([2.0, 2.0, 0.0])
        metrics = masked_metrics(prediction, target, null_value=0.0)
        self.assertTrue(np.isclose(metrics["MAE"], 1.5))
        self.assertTrue(np.isclose(metrics["RMSE"], np.sqrt(2.5)))


if __name__ == "__main__":
    unittest.main()
