"""Validation scheduling and training state restoration."""
import copy
import math
import random

import numpy as np
import torch

from pieps.runtime import OBJECTIVE

TRAINER = "single_stage_coarse_aux_plateau_v1"


def validate_training_config(training):
    expected = {"epochs", "seed", "learning_rate", "weight_decay", "grad_clip", "eval_every",
                "time_prefix_ratio", "activation_checkpointing", "scheduler", "early_stopping", "coarse_loss_weight"}
    if not isinstance(training, dict) or set(training) != expected:
        raise ValueError(f"Training configuration must specify {sorted(expected)}")

    def integer(value, name, minimum):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    def number(value, name, positive=True):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")

    number(training["coarse_loss_weight"], "coarse_loss_weight", positive=False)
    integer(training["epochs"], "epochs", 1)
    integer(training["seed"], "seed", 0)
    if training["seed"] >= 2**32:
        raise ValueError("seed must be less than 2**32")
    if type(training["eval_every"]) is not int or training["eval_every"] != 1:
        raise ValueError("Plateau scheduling requires eval_every=1; patience is measured in epochs")
    if type(training["activation_checkpointing"]) is not bool:
        raise ValueError("activation_checkpointing must be true or false")
    for key in ("learning_rate", "grad_clip", "time_prefix_ratio"):
        number(training[key], key)
    number(training["weight_decay"], "weight_decay", positive=False)
    if training["time_prefix_ratio"] > 1:
        raise ValueError("time_prefix_ratio must be <= 1")
    sc = training["scheduler"]
    required = {"name", "min_lr", "factor", "patience", "cooldown", "threshold", "window"}
    if not isinstance(sc, dict) or set(sc) != required or sc["name"] != "plateau":
        raise ValueError(f"Scheduler must be plateau and specify {sorted(required)}")
    for key in ("min_lr", "factor", "threshold"):
        number(sc[key], key)
    if sc["min_lr"] > training["learning_rate"] or sc["factor"] >= 1:
        raise ValueError("Require 0 < min_lr <= learning_rate and 0 < factor < 1")
    for key in ("patience", "cooldown", "window"):
        integer(sc[key], key, 1 if key == "window" else 0)
    early = training["early_stopping"]
    if not isinstance(early, dict) or set(early) != {"patience"}:
        raise ValueError("early_stopping must specify patience")
    integer(early["patience"], "early_stopping.patience", 1)


class ValidationSchedule:
    """Smooth validation for scheduling; select checkpoints using unsmoothed MAE."""

    def __init__(self, optimizer, training):
        self.optimizer = optimizer
        self.config = copy.deepcopy(training["scheduler"])
        self.early_patience = training["early_stopping"]["patience"]
        sc = self.config
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=sc["factor"], patience=sc["patience"],
            threshold=sc["threshold"], threshold_mode="abs", cooldown=sc["cooldown"],
            min_lr=sc["min_lr"], eps=0.)
        self.values = []
        self.floor_best = None
        self.floor_bad_epochs = 0
        self.stopped = False

    def step(self, validation):
        if not math.isfinite(validation):
            raise ValueError("Validation metric must be finite")
        if self.stopped:
            raise ValueError("Training already stopped")
        self.values = (self.values + [float(validation)])[-self.config["window"]:]
        smoothed = None
        # The epoch that lowers LR to the floor did not TRAIN at the floor yet.
        trained_at_floor = all(g["lr"] <= self.config["min_lr"] for g in self.optimizer.param_groups)
        if len(self.values) == self.config["window"]:
            smoothed = sum(self.values) / len(self.values)
            self.scheduler.step(smoothed)
            if trained_at_floor:
                if self.floor_best is None or smoothed < self.floor_best - self.config["threshold"]:
                    self.floor_best, self.floor_bad_epochs = smoothed, 0
                else:
                    self.floor_bad_epochs += 1
                self.stopped = self.floor_bad_epochs >= self.early_patience
        return {"validation_smoothed_mae_deg": smoothed,
                "floor_bad_epochs": self.floor_bad_epochs, "stopped_early": self.stopped}

    def state_dict(self):
        return {"kind": TRAINER, "config": copy.deepcopy(self.config),
                "early_patience": self.early_patience, "scheduler": self.scheduler.state_dict(),
                "values": list(self.values), "floor_best": self.floor_best,
                "floor_bad_epochs": self.floor_bad_epochs, "stopped": self.stopped}

    def load_state_dict(self, state):
        if (state.get("kind") != TRAINER or state["config"] != self.config
                or state["early_patience"] != self.early_patience):
            raise ValueError("Scheduler configuration differs from the checkpoint")
        self.scheduler.load_state_dict(state["scheduler"])
        self.values = list(state["values"])
        self.floor_best = state["floor_best"]
        self.floor_bad_epochs = state["floor_bad_epochs"]
        self.stopped = state["stopped"]


def capture_rng(device):
    return {"rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None}


def restore_rng(payload, device):
    random.setstate(payload["rng_python"])
    np.random.set_state(payload["rng_numpy"])
    torch.set_rng_state(payload["rng_torch"])
    if torch.device(device).type == "cuda":
        torch.cuda.set_rng_state(payload["rng_cuda"], device)




def restore_training_state(model, optimizer, source, device, learning_rate=None):
    model.load_state_dict(source["model"], strict=True)
    optimizer.load_state_dict(source["optimizer"])
    if learning_rate is not None:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
            if "initial_lr" in group:
                group["initial_lr"] = learning_rate
    restore_rng(source, device)
