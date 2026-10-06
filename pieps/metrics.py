"""Normal-map metrics and mask filling shared by inference and evaluation."""
import numpy as np


def build_dense_pred_map(pred, pixel_idx, h, w):
    pred_map = np.zeros((h * w, 3), dtype=np.float32)
    pred_map[pixel_idx] = pred.cpu().numpy().astype(np.float32)
    return pred_map.reshape(h, w, 3)


def _neighbor_sum_and_count(values, valid_mask):
    h, w, _ = values.shape
    padded_val = np.pad(values, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=0.0)
    padded_valid = np.pad(valid_mask.astype(np.float32), ((1, 1), (1, 1)), mode="constant", constant_values=0.0)
    sum_val = np.zeros((h, w, 3), dtype=np.float32)
    sum_cnt = np.zeros((h, w), dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            sum_val += padded_val[dy : dy + h, dx : dx + w, :]
            sum_cnt += padded_valid[dy : dy + h, dx : dx + w]
    return sum_val, sum_cnt


def fill_mask_holes(pred_map, mask):
    pred_map = np.asarray(pred_map, dtype=np.float32).copy()
    mask_bool = np.asarray(mask) > 0
    norm = np.linalg.norm(pred_map, axis=-1)
    valid = mask_bool & (norm > 1e-6)
    holes = mask_bool & (~valid)
    if not holes.any():
        return pred_map

    max_iters = max(pred_map.shape[0], pred_map.shape[1])
    for _ in range(max_iters):
        holes = mask_bool & (~valid)
        if not holes.any():
            break
        sum_val, sum_cnt = _neighbor_sum_and_count(pred_map, valid)
        fillable = holes & (sum_cnt > 0)
        if not fillable.any():
            break
        avg = sum_val[fillable] / sum_cnt[fillable, None]
        avg_norm = np.linalg.norm(avg, axis=-1, keepdims=True)
        avg = avg / np.maximum(avg_norm, 1e-8)
        pred_map[fillable] = avg.astype(np.float32)
        valid[fillable] = True

    holes = mask_bool & (~valid)
    if holes.any():
        valid_coords = np.argwhere(valid)
        hole_coords = np.argwhere(holes)
        if valid_coords.size > 0:
            for hy, hx in hole_coords:
                dist2 = np.sum((valid_coords - np.array([hy, hx])) ** 2, axis=1)
                ny, nx = valid_coords[int(np.argmin(dist2))]
                pred_map[hy, hx] = pred_map[ny, nx]
            valid[holes] = True

    pred_map[~mask_bool] = 0.0
    return pred_map.astype(np.float32)


def dense_angle_mae(pred_map, gt_map, mask):
    mask_bool = np.asarray(mask) > 0
    if not mask_bool.any():
        return float("nan")
    pred = pred_map[mask_bool]
    gt = gt_map[mask_bool]
    pred = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-8)
    gt = gt / np.maximum(np.linalg.norm(gt, axis=1, keepdims=True), 1e-8)
    cos = np.sum(pred * gt, axis=1)
    cos = np.clip(cos, -1.0 + 1e-7, 1.0 - 1e-7)
    ang = np.arccos(cos) * (180.0 / np.pi)
    return float(np.mean(ang))
