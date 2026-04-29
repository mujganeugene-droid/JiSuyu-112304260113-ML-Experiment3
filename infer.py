import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageOps

from model_defs import StrongCnn


MEAN = 0.1307
STD = 0.3081
CLASS_NAMES = [str(i) for i in range(10)]

PLAIN_VIEWS = [(0, 0)]
PLUS_VIEWS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
GRID_VIEWS = PLUS_VIEWS + [(1, 1), (1, -1), (-1, 1), (-1, -1)]

BEST_ENSEMBLE = [
    {"name": "rescnn_s1337_w32_a11.pt", "mode": "plus", "weight": 0.09218302},
    {"name": "rescnn_s2026_w40_a09.pt", "mode": "grid", "weight": 0.40695877},
    {"name": "rescnn_s42_w32_a10.pt", "mode": "grid", "weight": 0.40787306},
    {"name": "rescnn_s7_w48_a08.pt", "mode": "grid", "weight": 0.09298515},
]
MODE_TO_VIEWS = {"plain": PLAIN_VIEWS, "plus": PLUS_VIEWS, "grid": GRID_VIEWS}


@dataclass
class LoadedModel:
    name: str
    mode: str
    weight: float
    model: StrongCnn


class EnsemblePredictor:
    def __init__(self, ckpt_dir: str = "advanced_ckpts"):
        self.ckpt_dir = ckpt_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = self._load_models()

    def _load_models(self) -> list[LoadedModel]:
        loaded: list[LoadedModel] = []
        for item in BEST_ENSEMBLE:
            ckpt_path = os.path.join(self.ckpt_dir, item["name"])
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"checkpoint missing: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            cfg = ckpt["config"]
            model = StrongCnn(width=int(cfg["width"]), dropout=float(cfg["dropout"])).to(self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            loaded.append(LoadedModel(name=item["name"], mode=item["mode"], weight=float(item["weight"]), model=model))
        return loaded

    def _pick_foreground_mask(self, arr: np.ndarray) -> np.ndarray:
        total = arr.size
        mean = float(arr.mean())
        std = float(arr.std()) + 1e-6
        dark_mask = arr < (mean - 0.35 * std)
        light_mask = arr > (mean + 0.35 * std)

        dark_count = int(dark_mask.sum())
        light_count = int(light_mask.sum())
        min_px = max(24, int(total * 0.002))
        max_px = int(total * 0.65)

        dark_ok = min_px <= dark_count <= max_px
        light_ok = min_px <= light_count <= max_px

        if dark_ok and light_ok:
            return dark_mask if dark_count <= light_count else light_mask
        if dark_ok:
            return dark_mask
        if light_ok:
            return light_mask
        return dark_mask if dark_count <= light_count else light_mask

    def _largest_component_mask(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        visited = np.zeros((h, w), dtype=np.uint8)
        components: list[tuple[int, int, int, int, int]] = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for y in range(h):
            for x in range(w):
                if not mask[y, x] or visited[y, x]:
                    continue
                queue = deque([(y, x)])
                visited[y, x] = 1
                area = 0
                y0 = y1 = y
                x0 = x1 = x
                while queue:
                    cy, cx = queue.popleft()
                    area += 1
                    y0 = min(y0, cy)
                    y1 = max(y1, cy)
                    x0 = min(x0, cx)
                    x1 = max(x1, cx)
                    for dy, dx in directions:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = 1
                            queue.append((ny, nx))
                components.append((area, y0, y1, x0, x1))

        if not components:
            return mask

        center_y = (h - 1) / 2.0
        center_x = (w - 1) / 2.0
        best_idx = 0
        best_score = -1e18
        for i, (area, y0, y1, x0, x1) in enumerate(components):
            comp_cy = (y0 + y1) / 2.0
            comp_cx = (x0 + x1) / 2.0
            dist2 = (comp_cy - center_y) ** 2 + (comp_cx - center_x) ** 2
            score = area - 0.08 * dist2
            if score > best_score:
                best_score = score
                best_idx = i

        _, y0, y1, x0, x1 = components[best_idx]
        out = np.zeros_like(mask, dtype=bool)
        out[y0 : y1 + 1, x0 : x1 + 1] = mask[y0 : y1 + 1, x0 : x1 + 1]
        return out

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image.convert("L"))
        arr = np.asarray(image, dtype=np.uint8)

        fg_mask = self._largest_component_mask(self._pick_foreground_mask(arr))
        self._last_mask_image = Image.fromarray((fg_mask.astype(np.uint8) * 255), mode="L")
        ys, xs = np.where(fg_mask)
        if ys.size == 0 or xs.size == 0:
            return Image.new("L", (28, 28), 0)

        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        pad = max(2, int(0.06 * max(y1 - y0, x1 - x0)))
        y0 = max(0, y0 - pad)
        y1 = min(arr.shape[0], y1 + pad)
        x0 = max(0, x0 - pad)
        x1 = min(arr.shape[1], x1 + pad)

        bg_exists = bool((~fg_mask).any())
        fg_mean = float(np.mean(arr[fg_mask]))
        bg_mean = float(np.mean(arr[~fg_mask])) if bg_exists else 255.0
        work = (255 - arr) if fg_mean < bg_mean else arr.copy()

        digit_arr = work[y0:y1, x0:x1]
        digit = Image.fromarray(digit_arr, mode="L")

        w, h = digit.size
        side = max(w, h)
        square = Image.new("L", (side, side), 0)
        square.paste(digit, ((side - w) // 2, (side - h) // 2))

        resized = square.resize((20, 20), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (28, 28), 0)
        canvas.paste(resized, (4, 4))

        arr2 = np.asarray(canvas, dtype=np.float32)
        coords = np.argwhere(arr2 > 8)
        if coords.size > 0:
            center_y, center_x = coords.mean(axis=0)
            shift_x = int(round(13.5 - center_x))
            shift_y = int(round(13.5 - center_y))
            canvas = ImageChops.offset(canvas, shift_x, shift_y)
        return canvas

    def image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)

    def _shift_tensor(self, x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        if dx == 0 and dy == 0:
            return x
        pad_left = max(dx, 0)
        pad_right = max(-dx, 0)
        pad_top = max(dy, 0)
        pad_bottom = max(-dy, 0)
        padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        x_start = max(-dx, 0)
        y_start = max(-dy, 0)
        return padded[:, :, y_start : y_start + x.size(2), x_start : x_start + x.size(3)]

    @torch.no_grad()
    def predict(self, image: Image.Image, use_extra_tta: bool = False) -> dict[str, Any]:
        started = time.perf_counter()
        image_l = image.convert("L")
        candidates = [
            ("normal", self.preprocess_image(image_l), getattr(self, "_last_mask_image", None)),
            ("inverted", self.preprocess_image(ImageOps.invert(image_l)), getattr(self, "_last_mask_image", None)),
        ]
        best_probs = None
        best_processed = None
        best_mask = None
        best_conf = -1.0

        for _, processed, mask_img in candidates:
            tensor = self.image_to_tensor(processed)
            prob_sum = None
            total_weight = 0.0
            for loaded in self.models:
                views = list(MODE_TO_VIEWS[loaded.mode])
                if use_extra_tta and loaded.mode == "plain":
                    views = PLUS_VIEWS
                elif use_extra_tta and loaded.mode == "plus":
                    views = GRID_VIEWS

                tta_prob = None
                for dx, dy in views:
                    logits = loaded.model(self._shift_tensor(tensor, dx, dy))
                    probs = F.softmax(logits, dim=1)
                    tta_prob = probs if tta_prob is None else tta_prob + probs
                tta_prob = tta_prob / len(views)

                weighted = tta_prob * loaded.weight
                prob_sum = weighted if prob_sum is None else prob_sum + weighted
                total_weight += loaded.weight

            cur_probs = (prob_sum / total_weight).squeeze(0).detach().cpu().numpy()
            cur_conf = float(np.max(cur_probs))
            if cur_conf > best_conf:
                best_conf = cur_conf
                best_probs = cur_probs
                best_processed = processed
                best_mask = mask_img

        probs = best_probs
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        top_indices = np.argsort(probs)[::-1][:3]
        top3 = pd.DataFrame(
            {
                "Rank": [1, 2, 3],
                "Digit": [CLASS_NAMES[idx] for idx in top_indices],
                "Confidence": [f"{probs[idx] * 100:.2f}%" for idx in top_indices],
            }
        )
        return {
            "prediction": CLASS_NAMES[int(top_indices[0])],
            "confidence": float(probs[top_indices[0]]),
            "top3": top3,
            "probs": probs,
            "processed_image": best_processed,
            "mask_image": best_mask if best_mask is not None else best_processed,
            "latency_ms": elapsed_ms,
            "extra_tta": use_extra_tta,
            "model_count": len(self.models),
            "ensemble_desc": "one_shot_best (fixed weighted ensemble)",
        }


def make_probability_plot(probs: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(CLASS_NAMES, probs, color="#0f766e")
    best = int(np.argmax(probs))
    bars[best].set_color("#f97316")
    ax.set_title("Probability Distribution")
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    return fig
