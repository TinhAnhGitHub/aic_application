from unilm.beit3 import modeling_finetune
from torchvision import transforms
from transformers import XLMRobertaTokenizer
from timm import create_model
import timm
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from sentence_transformers import SentenceTransformer
import os
from typing import List
import numpy as np
import torch
from PIL import Image



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float32


class Processor():
    def __init__(self, tokenizer):
        self.image_processor = transforms.Compose([
            transforms.Resize((384, 384), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])
        
        self.tokenizer = tokenizer
    
    def process(self, image=None, text=None):
        assert (image is not None) or (text is not None)
        language_tokens = None
        padding_mask = None
        if image is not None:
            image = self.image_processor(image)
            image = image.unsqueeze(0)
        if text is not None:
            language_tokens, padding_mask, _ = self.get_text_segment(text)
        return {'image': image, 'text_description': language_tokens, 'padding_mask': padding_mask}
            
        
    def get_text_segment(self, text, max_len=64):
        tokens = self.tokenizer.tokenize(text)
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.tokenizer.bos_token_id] + tokens[:] + [self.tokenizer.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        language_tokens = tokens + [self.tokenizer.pad_token_id] * (max_len - num_tokens)
        return torch.tensor([language_tokens]),  torch.tensor([padding_mask]), num_tokens
    
    def process_batch(self, images):
        batch_images = [self.image_processor(img) for img in images]
        batch_images = torch.stack(batch_images, dim=0)  
        return  batch_images
    


class ModelService:
    """Singleton service for BEiT3 (image) + SentenceTransformer (text) embeddings"""

    def __init__(
        self,
        beit3_ckpt: str,
        beit3_tokenizer_path: str,
        text_model_name: str = "AITeamVN/Vietnamese_Embedding",
    ):
        self.device = DEVICE
        self._init_text(text_model_name)
        self._init_vision(beit3_ckpt, beit3_tokenizer_path)

    def _init_text(self, model_name: str):
        model = SentenceTransformer(model_name, device=self.device)
        model.max_seq_length = 2048
        model.eval()
        self.text_model = model

    def _init_vision(self, ckpt_path: str, tokenizer_path: str):
        self.vision_model = create_model("beit3_large_patch16_384_retrieval")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.vision_model.load_state_dict(ckpt["model"], strict=True)
        self.vision_model.to(self.device, dtype=TORCH_DTYPE).eval()

        self.tokenizer = XLMRobertaTokenizer(tokenizer_path)
        self.processor = Processor(self.tokenizer)
    
    @torch.inference_mode()
    def embed_text(self, text: str) -> list[float]:
        embs = self.text_model.encode(
            [text],
            batch_size=1,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=self.device,
        )
        embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
        return embs[0].detach().cpu().numpy().astype(np.float32)
        

    @torch.inference_mode()
    def embed_images(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i+batch_size]
            inputs = self.processor.process_batch(batch_imgs).to(self.device, dtype=TORCH_DTYPE)
            img_feat, _ = self.vision_model(image=inputs, only_infer=True)
            img_feat = torch.nn.functional.normalize(img_feat, p=2, dim=-1)
            all_feats.append(img_feat.detach().cpu())

        if not all_feats:
            return np.zeros((0, 0), dtype=np.float32)

        return torch.cat(all_feats, dim=0).numpy().astype(np.float32)

