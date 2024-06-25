import os
from typing import Optional

from enum import Enum

import torch

from modeling.imagebind.models.imagebind_model import ImageBindModel
from modeling.imagebind.models.imagebind_model import (
    ModalityType as ImageBindModalityType,
)
from modeling.imagebind.models.imagebind_model import imagebind_huge
from modeling.multimodal.encoders.image.base import BaseImageEncoder
from modeling.multimodal.encoders.registry import ModalityEncoderRegistry

from settings import ModalityEncoderType


@ModalityEncoderRegistry.register(ModalityEncoderType.IMAGEBIND_IMAGE)
class ImageBindImageModeling(BaseImageEncoder):
    def __init__(self, *args, imagebind_model: Optional[ImageBindModel] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if imagebind_model is not None:
            self.model_imagebind = imagebind_model
        else:
            self.model_imagebind = imagebind_huge(
                pretrained=int(os.getenv('IMAGEBIND_PRETRAINED', '1'))
            ).half()  # FIXME: For deepspeed

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        input_values = {
            ImageBindModalityType.VISION: inputs,
        }
        image_embeddings = self.model_imagebind(input_values)[ImageBindModalityType.VISION]
        image_embeddings = torch.nn.functional.normalize(image_embeddings, dim=-1)
        return image_embeddings

    @property
    def emb_dim(self) -> int:
        return self.model_imagebind.modality_heads[ImageBindModalityType.VISION][-1].out_features
