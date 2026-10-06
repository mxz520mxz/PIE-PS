"""Input separation, split integrity, and final-loss gradient connectivity."""
import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pieps.data import INPUT_KEYS, load_scene, model_inputs
from pieps.runtime import build_model, final_normal_loss
from pieps.splits import load_splits


def example_scene(offset=0):
    rng = np.random.default_rng(8 + offset)
    pixels = np.repeat(np.arange(400, 416), 6)
    lights = rng.normal(size=(len(pixels), 3)).astype(np.float32)
    lights[:, 2] = np.abs(lights[:, 2]) + 0.4
    lights /= np.linalg.norm(lights, axis=1, keepdims=True)
    next_lights = np.roll(lights, 1, axis=0).copy()
    e = np.zeros((len(pixels), 5), dtype=np.float32)
    e[:, 0], e[:, 3], e[:, 4] = pixels, rng.uniform(size=len(pixels)), rng.normal(size=len(pixels))
    mask = np.zeros((256, 256), dtype=np.float32)
    mask.flat[np.unique(pixels)] = 1
    gt = np.zeros((256, 256, 3), dtype=np.float32)
    gt[:, :, 2] = 1
    return dict(H=256, W=256, E=e, L_k=lights, L_k1=next_lights,
                p_k1=np.where(np.arange(len(pixels)) % 2, 1, -1).astype(np.float32), mask=mask, n_gt=gt)


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, name, value):
        path = self.root / name
        with path.open("wb") as stream:
            pickle.dump(value, stream)
        return path

    def test_inputs_do_not_depend_on_targets_or_threshold(self):
        scene = example_scene()
        original = load_scene(self.save("original.pkl", scene), require_target=True)
        changed = copy.deepcopy(scene)
        changed["n_gt"] = -changed["n_gt"]
        changed["C"] = 1000
        other = load_scene(self.save("changed.pkl", changed))
        del changed["n_gt"]
        del changed["C"]
        without = self.save("without.pkl", changed)
        independent = load_scene(without)
        for key in INPUT_KEYS:
            self.assertTrue(torch.equal(original[key], other[key]), key)
            self.assertTrue(torch.equal(original[key], independent[key]), key)
        self.assertNotIn("pixel_n_gt", independent)
        with self.assertRaises(ValueError):
            load_scene(without, require_target=True)

    def test_split_rejects_copied_scene(self):
        a = self.save("a.pkl", example_scene())
        (self.root / "b.pkl").write_bytes(a.read_bytes())
        self.save("c.pkl", example_scene(1))
        manifest = self.root / "splits.json"
        manifest.write_text(json.dumps(dict(train=["a.pkl"], validation=["b.pkl"], test=["c.pkl"])))
        with self.assertRaisesRegex(ValueError, "Duplicate scene content"):
            load_splits(manifest, self.root)

    def test_final_loss_rejects_empty_supervision(self):
        with self.assertRaises(ValueError):
            final_normal_loss(torch.ones(3, 3), torch.ones(3, 3), torch.zeros(3, dtype=torch.bool))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA graph extension required")
    def test_final_loss_reaches_all_parameter_groups(self):
        torch.manual_seed(42)
        model = build_model({}, torch.device("cuda:0"))
        batch = load_scene(self.save("scene.pkl", example_scene()), require_target=True)
        pred = model(*model_inputs(batch, "cuda:0"))
        loss = final_normal_loss(pred, batch["pixel_n_gt"].cuda(), batch["pixel_valid"].cuda())
        loss.backward()
        groups = ("conv_block1", "raw_mlp", "gnn_mlp", "proj", "pixel_convs", "pixel_norms", "normal_head",
                  "scorer_mlp", "scorer_layers", "scorer_head", "refine_proj", "refine_pixel_convs",
                  "refine_pixel_norms", "refine_normal_head")
        for name in groups:
            params = list(getattr(model, name).parameters())
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params), name)
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in params), 0, name)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA graph extension required")
    def test_scorer_recomputation_preserves_values_and_gradients(self):
        torch.manual_seed(42)
        model = build_model({"scorer_pixel_chunk": 8}, torch.device("cuda:0"))
        scores = torch.randn(19, 6, 80, device="cuda:0", requires_grad=True)
        mask = torch.ones(19, 6, device="cuda:0")
        mask[::2, -2:] = 0
        # Compare the changed memory-management path directly, avoiding unrelated
        # nondeterministic graph scatter reductions in this equivalence check.
        original = model._score_observations(scores, mask)[1]
        original.square().sum().backward()
        input_grad = scores.grad.detach().clone()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        scores.grad = None
        model.checkpoint_scorer = True
        actual = model._score_observations(scores, mask)[1]
        actual.square().sum().backward()
        torch.testing.assert_close(actual, original, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(scores.grad, input_grad, atol=1e-5, rtol=1e-4)
        for name, param in model.named_parameters():
            if name in grads:
                torch.testing.assert_close(param.grad, grads[name], atol=1e-4, rtol=1e-4, msg=name)


if __name__ == "__main__":
    unittest.main()
