import torch
from typing import List, Dict, Tuple, Optional

from inference_loader import InferenceMultimodalLoader
from modeling.imagebind import data, ModalityType
from transformers import PreTrainedTokenizer

from modeling.imagebind.models.imagebind_model import ImageBindModel
from settings import Modality
import torch.functional as F
import numpy as np

modality_tensor_cache = {}


def get_query_from_input(
    model,
    tokenizer: PreTrainedTokenizer,
    input_list: List[Dict[str, str]],
    experiment_config: Dict,
) -> Tuple[torch.LongTensor, List[Tuple[Modality, torch.Tensor]], torch.LongTensor]:
    all_text = ""
    modality_inputs = []
    modality_tokens = {experiment_config["img_token"], experiment_config["audio_token"]}

    modality_encoders = model.encoders


    def tokenize(text):
        return tokenizer(text, return_tensors="np", add_special_tokens=False)["input_ids"][0]


    audio_text_replica = experiment_config["modality_start"]
    audio_text_replica += experiment_config["audio_token"] * experiment_config["n_modality_embs"]
    audio_text_replica += experiment_config["modality_end"]
    audio_replica_tokens = tokenize(audio_text_replica)


    image_text_replica = experiment_config["modality_start"]
    image_text_replica += experiment_config["img_token"] * experiment_config["n_modality_embs"]
    image_text_replica += experiment_config["modality_end"]
    image_replica_tokens = tokenize(image_text_replica)


    prompt_begin_user_tokens = tokenize(experiment_config["prompt_begin_user"])
    prompt_begin_bot_tokens = tokenize(experiment_config["prompt_begin_bot"])
    prompt_begin_system_tokens = tokenize(experiment_config["prompt_begin_system"])
    prompt_end_tokens = tokenize(experiment_config["prompt_end_replica"])


    tokenized_replica = np.array([])

    system_prompt = np.concatenate((prompt_begin_system_tokens, tokenize(experiment_config["system_prompt"]), prompt_end_tokens))
    tokenized_replica = np.concatenate((tokenize(experiment_config["bos"]), system_prompt))

    tokenized_modality_tokens = []
    for t in (experiment_config["img_token"], experiment_config["audio_token"]):
        tt = tokenize(t)
        assert len(tt) == 1
        tokenized_modality_tokens.append(tt[0])

    for el in input_list:
        tokenized_replica = np.concatenate((tokenized_replica, prompt_begin_user_tokens))
        
        if el["type"] == "text":
            tokenized_replica = np.concatenate((tokenized_replica, tokenize(el["content"])))
        elif el["type"] == "image":
            # image_path = f"{experiment_config['image_path']}{el['content']}"
            image_path = el['content']
            # print(image_path)

            # if image_path in modality_tensor_cache:
            #     image_tensor = modality_tensor_cache[image_path]
            # else:
            #     image_tensor = load_images(modality_encoders[Modality.IMAGE], [image_path], device=experiment_config["device"])[0]
            #     modality_tensor_cache[image_path] = image_tensor
            image_tensor = load_images(modality_encoders[Modality.IMAGE], [image_path], device=experiment_config["device"])[0]

            modality_inputs.append((Modality.IMAGE, image_tensor))

            tokenized_replica = np.concatenate((tokenized_replica, image_replica_tokens))

        elif el["type"] == "audio":
            audio_path = f"{experiment_config['audio_path']}{el['content']}"

            if audio_path in modality_tensor_cache:
                audio_tensor = modality_tensor_cache[audio_path]
            else:
                audio_tensor = load_audios(modality_encoders[Modality.AUDIO], [audio_path], device=experiment_config["device"])[0]
                modality_tensor_cache[audio_path] = audio_tensor

            modality_inputs.append((Modality.AUDIO, audio_tensor))
            
            tokenized_replica = np.concatenate((tokenized_replica, audio_replica_tokens))

        tokenized_replica = np.concatenate((tokenized_replica, prompt_end_tokens))

    tokenized_replica = np.concatenate((tokenized_replica, prompt_begin_bot_tokens))

    modality_tokens_mask = np.isin(tokenized_replica, tokenized_modality_tokens).astype(np.int32)

    return (
        torch.LongTensor(tokenized_replica).to(experiment_config["device"]),
        modality_inputs,
        torch.LongTensor(modality_tokens_mask).to(experiment_config["device"])
    )

@torch.no_grad()
def load_audios(model_imagebind, audio_paths: List[str], device: torch.device):
    if isinstance(audio_paths, str):
        audio_paths = [audio_paths]
    inp = data.load_and_transform_audio_data(audio_paths=audio_paths, device=device)    
    universal_embeddings = model_imagebind.encode(inp).to(device)
    return universal_embeddings


@torch.no_grad()
def load_images(model_imagebind, image_paths: List[str], device: torch.device):
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    inp = data.load_and_transform_vision_data(image_paths, device)
    universal_embeddings = model_imagebind.encode(inp).to(device)
    return universal_embeddings

