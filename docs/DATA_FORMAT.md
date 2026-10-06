# Scene format

Each scene is a pickle dictionary with the following fields:

| Field | Shape | Description |
|---|---|---|
| `H`, `W` | Scalars | Image height and width; both 256 |
| `mask` | H × W | Positive values identify object pixels |
| `E` | N × at least 5 | Column 0: pixel index `y * W + x`; column 3: timestamp; column 4: PIE rate |
| `L_k`, `L_k1` | N × 3 | Paired lighting directions, aligned with observations |
| `p_k1` | N | Event polarity |
| `n_gt` | H × W × 3 | Ground-truth normals; required for training and evaluation |

Lighting directions and normals must use the same coordinate system. Ground-truth
normals must be finite and nonzero on object pixels. Predicted normals are unit
vectors with a nonnegative z component.

The loader sorts observations by pixel and timestamp, normalizes timestamps per
scene, scales PIE rates to [-1, 1] when their range is nonzero, and masks padding.
Pixel positions are represented as `(x / W, y / H)`. Model inputs are prepared from
observations independently of ground-truth normals. The loader does not require a
known contrast threshold or observation reliability labels.
