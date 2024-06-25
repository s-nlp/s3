from abc import ABC
from typing import Dict

from torch.nn import Module
from transformers import OPTForCausalLM, PreTrainedModel

from modeling.multimodal.encoders.base import BaseModalityEncoder

from settings import Modality


class BaseMultiModalModeling(Module, ABC):
    def __init__(
        self,
        language_model: PreTrainedModel,
        n_modality_embs: int,
        peft: bool = True,
    ) -> None:
        super().__init__()

        self.language_model = language_model

        self.n_modality_embs = n_modality_embs

        self.peft = peft

        print("Language model loaded")
        print(language_model)

        if self.peft:
            if isinstance(language_model.base_model.model, OPTForCausalLM):
                self.language_model_dim = (
                    language_model.base_model.model.model.decoder.embed_tokens.modules_to_save.default.weight.shape[1]
                )
            else:
                self.language_model_dim = (
                    language_model.base_model.model.model.embed_tokens.modules_to_save.default.weight.shape[1]
                )
        else:
            if isinstance(language_model, OPTForCausalLM):
                self.language_model_dim = language_model.model.decoder.embed_tokens.weight.shape[1]
            else:
                self.language_model_dim = language_model.model.embed_tokens.weight.shape[1]

