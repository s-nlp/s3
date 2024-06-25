from typing import Dict, Union
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from modeling.multimodal.encoders.base import BaseModalityEncoder
from modeling.multimodal.lm.fromage import FromageMultiModalModeling

import torch
from modeling.imagebind.models.imagebind_model import imagebind_huge
from modeling.multimodal.encoders.audio.imagebind import ImageBindAudioModeling
from modeling.multimodal.encoders.image.imagebind import ImageBindImageModeling

from settings import Modality, ModalityEncoderType

from transformers import AutoModelForCausalLM

from peft import PeftModel


def load_model(
    experiment_config: Dict,
    tokenizer: PreTrainedTokenizerBase,
) -> PreTrainedModel:

    model = AutoModelForCausalLM.from_pretrained(
        experiment_config["base_llm_path"]
    ).to(experiment_config["device"])

    model.resize_token_embeddings(len(tokenizer))

    if "peft_path" in experiment_config:
        model = PeftModel.from_pretrained(
            model,
            experiment_config["peft_path"],
            device_map="auto",
        )

    return model.to(experiment_config["device"])


class InferenceMultimodalLoader:
    # Read inference config and load models
    @staticmethod
    def load_modality_encoders(
        experiment_config: dict
    ) -> Dict[Modality, BaseModalityEncoder]:
        encoders_dict: Dict[Modality, BaseModalityEncoder] = {}

        modality_encoder_mapping = experiment_config['modality_encoder_mapping']
        device = experiment_config['device']

        # TODO: make configurable encoders
        imagebind_model = imagebind_huge(pretrained=True, path=experiment_config["imagebind_path"])
        # imagebind_model = imagebind_huge(pretrained=True)

        encoders_dict[Modality.IMAGE] = ImageBindImageModeling(
            imagebind_model=imagebind_model,
        ).to(device)
        encoders_dict[Modality.AUDIO] = ImageBindAudioModeling(
            imagebind_model=imagebind_model
        ).to(device)

        return encoders_dict

    @staticmethod
    def load_model(
        experiment_config: Dict, tokenizer: PreTrainedTokenizerBase
    ) -> Union[torch.nn.Module, PreTrainedModel]:
        language_model = load_model(
            experiment_config,
            tokenizer,
        )

        encoders = InferenceMultimodalLoader.load_modality_encoders(experiment_config)

        model = FromageMultiModalModeling(
            language_model=language_model,
            n_modality_embs=experiment_config["n_modality_embs"],
            peft=True,
            encoders=encoders,
        )

        model.modality_adapters.load_state_dict(torch.load(experiment_config["adapters_path"]))
 
        return model
