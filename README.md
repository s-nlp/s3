# S³: A Simple Strong Sample-effective Multimodal Dialog System

Elisei Rykov<sup>1</sup>, Egor Malkershin<sup>1</sup>, and Alexander Panchenko<sup>1,2</sup>

<sup>1</sup> Skolkovo Institute of Science and Technology, Russia
<sup>2</sup> Artificial Intelligence Research Institute, Russia
{e.rykov, egor.malkershin, a.panchenko}@skol.tech

### Abstract
In this work, we present a conceptually simple yet powerful baseline for the multimodal dialog task, an S<sup>3</sup> model, that achieves near state-of-the-art results on two compelling leaderboards: MMMU and AI Journey Contest 2023. The system is based on a pre-trained large language model, pre-trained modality encoders for image and audio, and a trainable modality projector. The proposed effective data mixture for training such an architecture demonstrates that a multimodal model based on a strong language model and trained on a small amount of multimodal data can perform efficiently in the task of multimodal dialog.

### Keywords: LLM, Multimodality, VQA, AQA

## Inference
First, download the LoRA adapter, tokenizer, imagebind, and projector weights from [s-nlp/s3](https://huggingface.co/s-nlp/s3). Then pass them to the generation function (see `example.ipynb` for details)