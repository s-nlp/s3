from enum import Enum


class Modality(str, Enum):
    IMAGE = 'image'
    AUDIO = 'audio'
    TEXT = 'text'


class ModalityEncoderType(str, Enum):
    IMAGEBIND_AUDIO = 'imagebind_audio'
    IMAGEBIND_IMAGE = 'imagebind_image'
    CLIP = 'clip'
