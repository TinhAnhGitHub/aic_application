from pydantic import BaseModel, Field
import numpy as np


class FusionWeights(BaseModel):
    w_visual: float = Field(default=0.5)
    w_caption: float = Field(default=0.3)
    w_ocr: float = Field(default=0.2)

    def ensure_scale(self) -> tuple[float,float,float]:
        """
        scale make sure total = 1
        """
        weights = np.array([self.w_visual, self.w_caption, self.w_ocr], dtype=float)
        min_val, max_val = weights.min(), weights.max()
        if min_val == max_val:
            normed = np.ones_like(weights) / len(weights)
        else:
            normed = (weights - min_val) / (max_val - min_val)
        s = float(normed.sum()) or 1.0
        normed = normed / s
        return float(normed[0]), float(normed[1]), float(normed[2])





class TopKReturn(BaseModel):
    topk_visual: int = Field(default=200)
    topk_caption: int = Field(default=200)
    topk_ocr: int = Field(default=400)
    final_topk: int = Field(default=300)


class ControllerParams(BaseModel):
    fusion: FusionWeights = Field(default_factory=FusionWeights)
    topk_settings: TopKReturn = Field(default_factory=TopKReturn)
    kf_search_param: dict = Field(default_factory=lambda: {"metric_type": "IP", "params": {"nprobe": 16}})
    cap_search_param: dict = Field(default_factory=lambda: {"metric_type": "IP", "params": {"nprobe": 16}})

    

