# S³: A Simple Strong Sample-effective Multimodal Dialog System

Elisei Rykov<sup>1</sup>, Egor Malkershin<sup>1</sup>, and Alexander Panchenko<sup>1,2</sup>

<sup>1</sup> Skolkovo Institute of Science and Technology, Russia
<sup>2</sup> Artificial Intelligence Research Institute, Russia
{e.rykov, egor.malkershin, a.panchenko}@skol.tech

Read the paper: https://arxiv.org/abs/2406.18305v1

### Abstract
In this work, we present a conceptually simple yet powerful baseline for the multimodal dialog task, an S<sup>3</sup> model, that achieves near state-of-the-art results on two compelling leaderboards: MMMU and AI Journey Contest 2023. The system is based on a pre-trained large language model, pre-trained modality encoders for image and audio, and a trainable modality projector. The proposed effective data mixture for training such an architecture demonstrates that a multimodal model based on a strong language model and trained on a small amount of multimodal data can perform efficiently in the task of multimodal dialog.

### Keywords: LLM, Multimodality, VQA, AQA

## Inference
First, download the LoRA adapter, tokenizer, imagebind, and projector weights from [s-nlp/s3](https://huggingface.co/s-nlp/s3). You should also pre-download [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) as it's a base model of S³. Then pass them to the generation function (see `example.ipynb` for details)

## Example
[See an example with image files on YouTube](https://youtu.be/eUH6LNCADKQ)

[See an example with audio files on YouTube](https://youtu.be/uTaOLo04LeM)

## Citation

```
@inproceedings{rykov-etal-2024-s3,
    title = "S$^3$: A Simple Strong Sample-effective Multimodal Dialog System",
    author = "Rykov, Elisei and Malkershin, Egor and Panchenko, Alexander",
    month = jun,
    year = "2024",
    address = "Turin, Italy",
    abstract = "In this work, we present a conceptually simple yet powerful baseline for multimodal dialog task, an S$^3$ model, that achieves near state-of-the-art results on two compelling leaderboards: MMMU and AI Journey Contest 2023. The system is based on a pre-trained large language model, pre-trained modality encoders for image and audio, and a trainable modality projector. The proposed effective data mixture for training such an architecture demonstrates that a multimodal model based on a strong language model and trained on a small amount of multimodal data can perform efficiently in the task of multimodal dialog.",
    conference = "Natural Language Processing and Information Systems"
}

```
