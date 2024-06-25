from abc import ABC

from modeling.multimodal.encoders.base import BaseModalityEncoder


class BaseAudioEncoder(BaseModalityEncoder, ABC):
    ...
