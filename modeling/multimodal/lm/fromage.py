from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from scipy.ndimage.measurements import find_objects, label
from transformers import OPTForCausalLM
from transformers.utils import ModelOutput

from modeling.multimodal.lm.base import BaseMultiModalModeling

from settings import Modality


class ModalityAdapter(torch.nn.Module):
    def __init__(
        self, language_model_dim: int, encoder_emb_dim: int, n_modality_embs: int, dropout_prob: float, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.linear_projection_1 = torch.nn.Linear(encoder_emb_dim, language_model_dim)
        self.linear_projection_2 = torch.nn.Linear(language_model_dim, language_model_dim * n_modality_embs)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, encoded_modality_object_batch: torch.Tensor):
        output = self.linear_projection_1(encoded_modality_object_batch)
        output = self.dropout(output)

        output = self.linear_projection_2(output)
        output = self.dropout(output)
        return output


class FromageMultiModalModeling(BaseMultiModalModeling):
    def __init__(self, encoders, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = self.language_model.device  # TODO: base class requested device from model
       
        self.encoders = encoders

        modality_adapters = {}
        for modality in self.encoders.keys():
            modality_adapters[modality] = ModalityAdapter(
                language_model_dim=self.language_model_dim,
                encoder_emb_dim=self.encoders[modality].emb_dim,
                n_modality_embs=self.n_modality_embs,
                dropout_prob=0.1,
            ).to(self.language_model.device)

        self.modality_adapters = torch.nn.ModuleDict(modality_adapters)

        self.config = self.language_model.config  # For deepspeed

    def convert_inputs_to_embeds(
        self,
        input_ids: torch.LongTensor,
        modality_inputs: List[List[Tuple[Modality, torch.Tensor]]],
        modality_tokens_mask: torch.Tensor,
    ) -> torch.Tensor:
        multimodal_lm_input_embeds: List[torch.Tensor] = []

        if self.peft:
            if isinstance(self.language_model.base_model.model, OPTForCausalLM):
                lm_input_embeds = (
                    self.language_model.base_model.model.model.decoder.embed_tokens.modules_to_save.default(input_ids)
                )
            else:  # LLaMA
                lm_input_embeds = self.language_model.base_model.model.model.embed_tokens.modules_to_save.default(
                    input_ids
                )
        else:
            if isinstance(self.language_model, OPTForCausalLM):
                lm_input_embeds = self.language_model.model.decoder.embed_tokens(input_ids)
            else:
                lm_input_embeds = self.language_model.model.embed_tokens(input_ids)

        for sample_lm_input_embeds, sample_modality_tokens_mask, sample_modality_inputs in zip(
            lm_input_embeds, modality_tokens_mask, modality_inputs
        ):
            span_mask, _ = label(
                sample_modality_tokens_mask.cpu().detach().numpy()
            )  # returns mask with ids of spans from 1 to N
            modality_spans = find_objects(span_mask)  # returns list of tuples with start index and end index

            grouped_modality_encoder_inputs: Dict[Modality, List[Tuple[int, torch.Tensor]]] = defaultdict(list)

            for index, modality_object in enumerate(sample_modality_inputs):
                modality, inputs = modality_object
                grouped_modality_encoder_inputs[modality].append((index, inputs))

            sorted_modality_embeddings: torch.Tensor = torch.full(
                (len(sample_modality_inputs), self.language_model_dim * self.n_modality_embs), torch.nan
            ).to(self.language_model.device)

            for modality, modality_encoder_inputs_with_indices in grouped_modality_encoder_inputs.items():
                modality_encoder_input_indexes, modality_encoder_inputs = zip(*modality_encoder_inputs_with_indices)

                modality_encoder_embeddings = self.modality_adapters[modality](torch.stack(modality_encoder_inputs, dim=0))

                sorted_modality_embeddings[modality_encoder_input_indexes, :] = modality_encoder_embeddings.to(
                    sorted_modality_embeddings.dtype
                )

            substituted_sample_lm_input_embeds = sample_lm_input_embeds.clone()
            for i, modality_span in enumerate(modality_spans):
                substituted_sample_lm_input_embeds[
                    modality_span[0].start : modality_span[0].stop
                ] = sorted_modality_embeddings[i, :].reshape(
                    self.n_modality_embs, self.language_model_dim
                )  # Split one projection into n_modality_embs projections

            multimodal_lm_input_embeds.append(substituted_sample_lm_input_embeds)

        return torch.stack(multimodal_lm_input_embeds)

    def forward(
        self,
        input_ids: torch.LongTensor,
        modality_inputs: List[List[Tuple[Modality, torch.Tensor]]],
        attention_mask: torch.LongTensor,
        modality_tokens_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,

    ) -> ModelOutput:
        multimodal_lm_input_embeds = self.convert_inputs_to_embeds(input_ids, modality_inputs, modality_tokens_mask)

        return self.language_model(
            inputs_embeds=multimodal_lm_input_embeds, labels=labels, attention_mask=attention_mask
        )
