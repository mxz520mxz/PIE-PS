"""Scheduling boundaries and optimizer/RNG state restoration."""
import copy
import random
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from pieps.runtime import OBJECTIVE
from pieps.training import (ValidationSchedule, capture_rng, restore_training_state, validate_training_config)


def recipe():
    path = Path(__file__).resolve().parents[1] / "configs/train.yaml"
    return yaml.safe_load(path.read_text())["training"]


def optimizer(lr=1e-4):
    return torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=lr)


class Scheduling(unittest.TestCase):
    def test_reductions_cooldown_floor_and_early_stop(self):
        tc = recipe()
        tc["scheduler"].update(window=2, patience=1, cooldown=1, min_lr=2.5e-5)
        tc["early_stopping"]["patience"] = 2
        opt = optimizer()
        control = ValidationSchedule(opt, tc)
        expected = [1e-4, 1e-4, 1e-4, 5e-5, 5e-5, 5e-5, 2.5e-5, 2.5e-5, 2.5e-5, 2.5e-5]
        for epoch, lr in enumerate(expected, 1):
            row = control.step(14.)
            self.assertAlmostEqual(opt.param_groups[0]["lr"], lr)
            self.assertEqual(row["stopped_early"], epoch == 10)
            if epoch == 7:
                self.assertEqual(control.floor_bad_epochs, 0)
                self.assertIsNone(control.floor_best)
        self.assertEqual(control.floor_bad_epochs, 2)

    def test_smoothing_ignores_single_good_epoch_and_real_progress_resets_stop(self):
        tc = recipe()
        tc["scheduler"].update(window=3, min_lr=1e-4)
        tc["early_stopping"]["patience"] = 3
        control = ValidationSchedule(optimizer(), tc)
        for value in (14., 14.):
            self.assertIsNone(control.step(value)["validation_smoothed_mae_deg"])
        self.assertEqual(control.step(11.)["validation_smoothed_mae_deg"], 13.)
        control.step(14.)
        self.assertEqual(control.floor_bad_epochs, 1)
        control.step(13.)
        self.assertEqual(control.floor_bad_epochs, 0)
        self.assertFalse(control.stopped)

    def test_resume_preserves_partial_window_cooldown_and_floor_counter(self):
        tc = recipe()
        tc["scheduler"].update(window=3, patience=1, cooldown=2, min_lr=2.5e-5)
        tc["early_stopping"]["patience"] = 30
        values = [14.] * 25
        for cut in (1, 5, 6, 14):
            with self.subTest(cut=cut):
                first_opt = optimizer()
                first = ValidationSchedule(first_opt, tc)
                for value in values[:cut]:
                    first.step(value)
                second_opt = optimizer()
                second_opt.load_state_dict(copy.deepcopy(first_opt.state_dict()))
                second = ValidationSchedule(second_opt, tc)
                second.load_state_dict(copy.deepcopy(first.state_dict()))
                for value in values[cut:]:
                    self.assertEqual(first.step(value), second.step(value))
                    self.assertEqual(first_opt.param_groups[0]["lr"], second_opt.param_groups[0]["lr"])
                self.assertEqual(first.state_dict(), second.state_dict())

    def test_training_budget_does_not_change_schedule(self):
        short, long = recipe(), recipe()
        short["epochs"], long["epochs"] = 50, 400
        a, b = ValidationSchedule(optimizer(), short), ValidationSchedule(optimizer(), long)
        for _ in range(50):
            self.assertEqual(a.step(14.), b.step(14.))
        self.assertEqual(a.state_dict(), b.state_dict())

    def test_config_rejects_zero_floor_stages_and_sparse_validation(self):
        tc = recipe()
        validate_training_config(tc)
        mutations = [("scheduler", "min_lr", 0), ("scheduler", "factor", 1),
                     ("scheduler", "threshold", float("nan")), ("scheduler", "window", True),
                     (None, "eval_every", 5), (None, "stages", {}), (None, "weight_decay", -1)]
        for section, key, value in mutations:
            with self.subTest(key=key):
                bad = copy.deepcopy(tc)
                (bad if section is None else bad[section])[key] = value
                with self.assertRaises(ValueError):
                    validate_training_config(bad)


class TrainingState(unittest.TestCase):
    @staticmethod
    def step(model, opt):
        # All three RNG streams contribute to the next Adam update.
        inputs = torch.randn(4, 2) * random.random() + float(np.random.random())
        opt.zero_grad()
        model(inputs).square().mean().backward()
        opt.step()

    def test_restoration_preserves_moments_and_next_random_update(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        model = torch.nn.Linear(2, 1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        for _ in range(4):
            self.step(model, opt)
        # Check restoration of Adam moments and random streams.
        opt.param_groups[0]["lr"] = 0.
        source = copy.deepcopy({"model": model.state_dict(), "optimizer": opt.state_dict(),
                                **capture_rng("cpu")})
        restored = torch.nn.Linear(2, 1)
        restored_opt = torch.optim.Adam(restored.parameters(), lr=1e-5)
        restore_training_state(restored, restored_opt, copy.deepcopy(source), "cpu", learning_rate=1e-5)
        self.assertEqual(restored_opt.param_groups[0]["lr"], 1e-5)
        for old, new in zip(source["optimizer"]["state"].values(), restored_opt.state_dict()["state"].values()):
            for key in ("step", "exp_avg", "exp_avg_sq"):
                self.assertTrue(torch.equal(old[key], new[key]), key)
        self.step(restored, restored_opt)
        result = copy.deepcopy(restored.state_dict())
        # Construct the reference without reinitializing Adam's moments.
        restore_training_state(model, opt, copy.deepcopy(source), "cpu", learning_rate=1e-5)
        self.step(model, opt)
        for key, value in result.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
        self.assertEqual(source["optimizer"]["param_groups"][0]["lr"], 0.)



if __name__ == "__main__":
    unittest.main()
