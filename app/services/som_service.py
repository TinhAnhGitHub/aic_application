from __future__ import annotations
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from datetime import datetime
import math
import json
import numpy as np

from app.schemas.search_results import KeyframeScore
from app.repository.som_repo import SomRepo


def gaussian_kernel(radius: int, sigma: float) -> np.ndarray:
    size = 2 * radius + 1
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma))
    s = k.sum()
    k = k / s if s > 0 else k
    return k.astype(np.float32)


@dataclass
class SomConfig:
    grid_h: int
    grid_w: int
    r: int = 2
    sigma: float = 1.2
    w_pos: float = 1.0
    w_neg: float = 0.4
    alpha: float = 1.0
    beta: float = 0.75
    gamma: float = 0.15
    kappa: float = 3.0


class SomFeedbackService:
    def __init__(self, bmu_map: Dict[int, Tuple[int, int]], repo: SomRepo, cfg: SomConfig):
        self.bmu_map = bmu_map
        self.repo = repo
        self.cfg = cfg
        self.kernel = gaussian_kernel(cfg.r, cfg.sigma)
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray, datetime]] = {}

    @staticmethod
    def load_bmu_map(path: str | None) -> Dict[int, Tuple[int, int]]:
        """
        # Allow {"123":[u,v], ...} formats
        """
        if not path:
            return {}
        data = json.load(open(path, 'r', encoding='utf-8'))
        return {int(k): tuple(v) for k, v in data.items()}
        


    async def _get_overlay(self, question: str) -> Tuple[np.ndarray, np.ndarray]:
        if question in self._cache:
            pos, neg = self._cache[question]
            return pos, neg
        pos, neg, doc = await self.repo.get_overlay(question, (self.cfg.grid_h, self.cfg.grid_w))
        self._cache[question] = (pos, neg)
        return pos, neg

    async def _save_overlay(self, question: str, pos: np.ndarray, neg: np.ndarray):
        _, _, doc = await self.repo.get_overlay(question, (self.cfg.grid_h, self.cfg.grid_w))
        await self.repo.save_overlay(question, pos, neg, doc)
        self._cache[question] = (pos, neg)

    async def apply_feedback(self, question: str, identification: int, action: str, weight: float = 1.0):
        bmu = self.bmu_map.get(int(identification))
        if bmu is None:
            return
        u, v = bmu
        pos, neg = await self._get_overlay(question)
        k = self.kernel
        r = self.cfg.r
        h, w = pos.shape
        u0, v0 = max(0, u - r), max(0, v - r)
        u1, v1 = min(h, u + r + 1), min(w, v + r + 1)
        ku0, kv0 = (0 if u - r >= 0 else r - u), (0 if v - r >= 0 else r - v)
        ku1, kv1 = (k.shape[0] - (u + r + 1 - u1)), (k.shape[1] - (v + r + 1 - v1))

        if action == "up":
            pos[u0:u1, v0:v1] += (self.cfg.w_pos * weight) * k[ku0:ku1, kv0:kv1]
        else:
            neg[u0:u1, v0:v1] += (self.cfg.w_neg * weight) * k[ku0:ku1, kv0:kv1]
        await self.repo.add_events(question, identification, 'up' if action == 'up' else 'down', weight)
        await self._save_overlay(question, pos, neg)

    @staticmethod
    def _sigmoid(x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    async def rerank_with_overlay(self, question: str, items: List[KeyframeScore], normalize_output: bool = True) -> List[KeyframeScore]:
        if not items:
            return items
        pos, neg = await self._get_overlay(question)
        alpha, beta, gamma, kappa = self.cfg.alpha, self.cfg.beta, self.cfg.gamma, self.cfg.kappa

        out: List[KeyframeScore] = []
        for it in items:
            s = float(it.score)
            bmu = self.bmu_map[int(it.identification)]
            u, v = bmu
            b_pos_raw = float(pos[u, v])
            b_neg_raw = float(neg[u, v])
            b_pos = self._sigmoid(kappa * b_pos_raw)
            b_neg = self._sigmoid(kappa * b_neg_raw)
            new_s = alpha * s + beta * b_pos - gamma * b_neg
            out.append(it.__class__(**{**it.model_dump(), "score": float(new_s)}))

        if not normalize_output:
            return sorted(out, key=lambda x: x.score, reverse=True)

        scores = [x.score for x in out]
        lo, hi = min(scores), max(scores)
        rng = (hi - lo) if hi - lo > 1e-6 else 1.0
        out2 = []
        for it in out:
            s = (it.score - lo) / rng
            out2.append(it.__class__(**{**it.model_dump(), "score": float(s)}))
        return sorted(out2, key=lambda x: x.score, reverse=True)

