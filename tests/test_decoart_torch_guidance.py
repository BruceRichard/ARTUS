import unittest

import torch

from experiments.decoart.torch_guidance import (
    guide_part_representations,
    physical_terms_for_candidates,
    route_candidates,
    tree_scores_for_candidates,
)


def _state(size, center, origin, axis, limit):
    return torch.tensor(size + center + origin + axis + limit, dtype=torch.float32)


class DecoArtTorchGuidanceTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "samples_per_face": 9,
            "dmax": 0.10,
            "alpha_normal": 0.50,
            "beta_penetration": 1.00,
            "tau_pen": 0.02,
            "theta_tan": 0.10,
            "theta_pen": 0.02,
            "lambda_depth": 0.55,
            "lambda_subtree": 0.45,
            "lambda_delta": 0.55,
            "lambda_omega": 0.45,
            "tau_structure": 0.50,
            "tau_physical": 0.35,
            "group_weights": {
                "structure": 0.50,
                "physical": 1.00,
                "detail": 0.25,
            },
            "routing_mode": "correct",
            "eta": 0.05,
            "steps": 3,
            "max_backtracks": 8,
        }
        self.start = torch.zeros(16)
        self.parent = _state(
            [2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        )
        self.existing = torch.stack((self.start, self.parent))
        self.existing_parents = torch.tensor([0, 0], dtype=torch.long)

    def test_physical_cost_detects_penetration(self):
        child = _state(
            [1.0, 1.0, 1.0],
            [-1.30, 0.0, 0.0],
            [-1.50, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
        ).unsqueeze(0)
        terms = physical_terms_for_candidates(
            child,
            self.existing,
            torch.tensor([1]),
            self.config,
        )
        self.assertGreater(float(terms["penetration"][0]), 0.0)
        self.assertGreater(float(terms["j_cost"][0]), 0.0)

    def test_bad_candidate_routes_to_physical(self):
        child = _state(
            [1.0, 1.0, 1.0],
            [-1.30, 0.0, 0.0],
            [-1.50, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
        ).unsqueeze(0)
        parent_indices = torch.tensor([1])
        terms = physical_terms_for_candidates(child, self.existing, parent_indices, self.config)
        structure = tree_scores_for_candidates(
            self.existing_parents,
            parent_indices,
            self.config,
        )
        routes = route_candidates(structure, terms, self.config)
        self.assertEqual(int(routes[0]), 1)

    def test_random_routing_preserves_group_counts(self):
        structure = torch.tensor([0.9, 0.1, 0.1, 0.8, 0.2])
        terms = {
            "align": torch.tensor([0.0, 0.5, 0.01, 0.0, 0.4]),
            "penetration": torch.tensor([0.0, 0.1, 0.0, 0.0, 0.2]),
        }
        correct = route_candidates(structure, terms, self.config, seed=7)
        random_config = dict(self.config, routing_mode="random")
        randomized = route_candidates(structure, terms, random_config, seed=7)
        self.assertEqual(
            torch.bincount(correct, minlength=3).tolist(),
            torch.bincount(randomized, minlength=3).tolist(),
        )

    def test_validity_vector_does_not_increase_cost(self):
        hidden = _state(
            [1.0, 1.0, 1.0],
            [-1.30, 0.0, 0.0],
            [-1.50, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
        ).unsqueeze(0)

        def decode(value):
            return {
                "articulated_info": value[:, :16],
                "condition": value.new_zeros((value.shape[0], 1)),
                "is_end_token_logits": value.new_ones((value.shape[0],)),
            }

        result = guide_part_representations(
            hidden,
            decode,
            self.existing,
            self.existing_parents,
            torch.tensor([1]),
            self.config,
        )
        self.assertLessEqual(result.log["j_post"], result.log["j_pre"] + 1.0e-6)
        self.assertGreater(result.log["accepted_steps"], 0)


if __name__ == "__main__":
    unittest.main()
