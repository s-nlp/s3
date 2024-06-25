from typing import Optional

import torch

from modeling.imagebind.models.imagebind_model import ImageBindModel
from modeling.imagebind.models.imagebind_model import (
    ModalityType as ImageBindModalityType,
)
from modeling.imagebind.models.imagebind_model import imagebind_huge
from modeling.multimodal.encoders.audio.base import BaseAudioEncoder
from modeling.multimodal.encoders.registry import ModalityEncoderRegistry
from settings import ModalityEncoderType


@ModalityEncoderRegistry.register(ModalityEncoderType.IMAGEBIND_AUDIO)
class ImageBindAudioModeling(BaseAudioEncoder):
    def __init__(self, imagebind_model: Optional[ImageBindModel] = None):
        super().__init__()

        if imagebind_model is not None:
            self.model_imagebind = imagebind_model
        else:
            self.model_imagebind = imagebind_huge(pretrained=True).half()  # FIXME: For deepspeed

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        input_values = {ImageBindModalityType.AUDIO: inputs}
        audio_embeddings = self.model_imagebind(input_values)[ImageBindModalityType.AUDIO]
        audio_embeddings = torch.nn.functional.normalize(audio_embeddings, dim=-1)
        return audio_embeddings

    @property
    def emb_dim(self) -> int:
        return self.model_imagebind.modality_heads[ImageBindModalityType.AUDIO][-1].out_features
