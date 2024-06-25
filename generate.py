import torch
from typing import Union, Optional, List, Tuple, Dict
from transformers import AutoTokenizer, PreTrainedTokenizer

from inference_loader import InferenceMultimodalLoader
from modeling.multimodal.lm.fromage import FromageMultiModalModeling
from utils import get_query_from_input

from torch import nn
import os


workdir = os.getcwd()


experiment_config = {
    "tokenizer_path": os.path.join(workdir, "team_code/ckpts/tokenizer/"),

    # MODEL SETTINGS
    "base_llm_path": "/ckpts/Mistral-7B-v0.1/",
    "imagebind_path": "/app/.checkpoints/imagebind_huge.pth",
    "peft_path": os.path.join(workdir, "team_code/ckpts/language_model/"),
    "adapters_path": os.path.join(workdir, "team_code/ckpts/projections/modality_adapters.pt"),

    "modality_encoder_mapping": {
        "image": "imagebind_image",
        "audio": "imagebind_audio"
    },
    "n_modality_embs": 4,

    "device": "cuda:0",
    "image_path": "",
    "audio_path": "",


    # PROMPT SETTINGS
    "prompt_begin_user": "<RS><user>",
    "prompt_begin_bot": "<RS><bot>",
    "prompt_begin_system": "<RS><system>",
    "system_prompt": "You are an AI assistant.",
    "prompt_end_replica": "</RS>",
    "modality_start": "<MS>",
    "modality_end": "</MS>",
    "img_token": "<img>",
    "audio_token": "<audio>",
    "bos": "<s>",
}

print("Generation config 🫡: ", experiment_config)


EOS_TOKEN = "</RS>"

gen_params = {
    "max_new_tokens": 32,
    "use_cache": True,
    "num_return_sequences": 1,
    # "num_beams": 3,
    # "do_sample": True
}


@torch.no_grad()
def gen_answer(
    model: FromageMultiModalModeling,
    tokenizer: PreTrainedTokenizer,
    context: Optional[torch.Tensor],
) -> str:
    if "eos_token_id" not in gen_params:
        gen_params['pad_token_id'] = tokenizer.pad_token_id
        gen_params['eos_token_id'] = tokenizer.encode(EOS_TOKEN, add_special_tokens=False)[0]

    out = model.language_model.generate(
        inputs_embeds=context.to(experiment_config["device"]),
        **gen_params,
    )

    generated_answer = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    if len(generated_answer) == 0:
        return "I can't answer that question"
    return generated_answer


def setup_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(experiment_config["tokenizer_path"], use_fast=False)
    model = InferenceMultimodalLoader.load_model(experiment_config, tokenizer)
    model.eval()
    return model, tokenizer


def update_history(
    model: FromageMultiModalModeling,
    tokenizer: PreTrainedTokenizer,
    history: torch.Tensor,
    previous_response_text: str,
    history_token_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    encoding = tokenizer.encode(
        previous_response_text,
        return_tensors="pt",
        add_special_tokens=False
    )

    end_replica_tokens = tokenizer.encode(
        experiment_config["prompt_end_replica"],
        return_tensors="pt",
        add_special_tokens=False,
    )

    encoding = torch.cat((encoding, end_replica_tokens), dim=1).to(experiment_config["device"])
    
    if history_token_ids is not None:
        history_token_ids = torch.cat([
            history_token_ids,
            encoding,
        ], dim=1)

    response_embedding = model.language_model.base_model.model.model.embed_tokens.modules_to_save.default(encoding)

    history = torch.concat([
        history,
        response_embedding,
    ], dim=1)

    return history, history_token_ids


def generate_text(
    model: FromageMultiModalModeling,
    tokenizer: PreTrainedTokenizer,
    cur_query_list: List[Dict[str, str]],
    history_list: Optional[Tuple[torch.Tensor, str]] = None
) -> Tuple[str, torch.Tensor]:
    """
    Args:
        cur_query_list: новая "реплика" юзера, которая может состоять из нескольких input-ов: картинка, аудио, текст
        history_list: (тензор с эмбеддингами диалога законченный предыдущим ПРОМПТОМ, предыдущий ответ)

    Returns:
        (текущий ответ модели, история с новым промптом без ответа модели)
    """

    history_tensor: Optional[torch.Tensor] = None
    history_token_ids: Optional[torch.Tensor] = None

    if history_list is not None:
        (history_tensor, history_token_ids), prev_answer = history_list
        history_tensor, history_token_ids = update_history(model, tokenizer, history_tensor, prev_answer, history_token_ids)

    # Кодируем новый запрос юзера, которой еще не было в history
    new_replica_input_ids, new_replica_modality_inputs, new_replica_modality_tokens_mask = get_query_from_input(
        model,
        tokenizer,
        cur_query_list,
        experiment_config,
        history_list is None,
    )

    # Делаем эмбеддинги для модели из нового запроса юзера
    inputs_embeds = model.convert_inputs_to_embeds(
        new_replica_input_ids.unsqueeze(0),
        [new_replica_modality_inputs],
        new_replica_modality_tokens_mask.unsqueeze(0),
    ).to(experiment_config["device"])


    if history_tensor is not None:
        history_tensor = torch.concat([
            history_tensor,
            inputs_embeds,
        ], dim=1)

        history_token_ids = torch.concat([
            history_token_ids,
            new_replica_input_ids.unsqueeze(0),
        ], dim=1)
    else:
        history_tensor = inputs_embeds
        history_token_ids = new_replica_input_ids.unsqueeze(0)

    # print(history_token_ids)

    # print(f"Model input: {tokenizer.batch_decode(history_token_ids, skip_special_tokens=False)}")

    # отвечаем на новый запрос с учетом history
    response_text = gen_answer(model, tokenizer, context=history_tensor)
    # print(f"Model answer: {response_text}")

    return response_text, (history_tensor, history_token_ids)


def get_ppl(
    model: FromageMultiModalModeling,
    tokenizer: PreTrainedTokenizer,
    cur_query_tuple: Tuple[List[Dict[str, str]], str],
    history: Optional[Tuple[torch.Tensor, str]] = None,
):
    """
    cur_query_tuple - (новый запрос который может состоять из нескольких реплик, ответ на котором нужно посчитать перплексию)
    history - (тензор с эмбеддингами диалога законченный предыдущим промптом, предыдущий ответ строкой)
    """

    history_tensor: Optional[torch.Tensor] = None

    if history is not None:
        history_tensor, prev_answer = history
        history_tensor, _ = update_history(model, tokenizer, history_tensor, prev_answer)

    # Кодируем новый запрос юзера, которой еще не было в history
    new_replica_input_ids, new_replica_modality_inputs, new_replica_modality_tokens_mask = get_query_from_input(
        model,
        tokenizer,
        cur_query_tuple[0],
        experiment_config,
        history is None,
    )

    # Делаем эмбеддинги для модели из нового запроса юзера
    inputs_embeds = model.convert_inputs_to_embeds(
        new_replica_input_ids.unsqueeze(0),
        [new_replica_modality_inputs],
        new_replica_modality_tokens_mask.unsqueeze(0),
    ).to(experiment_config["device"])

    if history_tensor is not None:
        dialogue_embeds = torch.cat([history_tensor, inputs_embeds], dim=1).to(experiment_config["device"])
    else:
        dialogue_embeds = inputs_embeds.to(experiment_config["device"])


    answer_ids = tokenizer(cur_query_tuple[1] + EOS_TOKEN,
                           return_tensors="pt", add_special_tokens=False).input_ids.to(experiment_config["device"])

    answer_embeds = model.language_model.base_model.model.model.embed_tokens.modules_to_save.default(answer_ids)
    
    inputs_embeds = torch.concat([dialogue_embeds, answer_embeds], dim=1)
    
    out_logits = model.language_model(
        inputs_embeds=inputs_embeds, attention_mask=torch.ones((1, inputs_embeds.shape[0])).to(experiment_config["device"]),
    ).logits

    shift_logits = out_logits[..., : -1, :].contiguous()
    context_before_labels = torch.LongTensor([-100] * dialogue_embeds.shape[1]).unsqueeze(0).to(experiment_config["device"])
    labels = torch.concat([context_before_labels, answer_ids], dim=1)
    shift_labels = labels[..., 1:].contiguous()
   
    loss = nn.CrossEntropyLoss()
    neg_log_likelihood = loss(shift_logits.transpose(1, 2), shift_labels)
    ppl = torch.exp2(neg_log_likelihood)
    
    return ppl.item(), dialogue_embeds

