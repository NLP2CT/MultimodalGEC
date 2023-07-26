# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BART model configuration"""
import warnings
from collections import OrderedDict
from typing import Any, Mapping, Optional

from typing import List, Union
from transformers import PretrainedConfig, logging, AutoConfig,CLIPVisionConfig
import copy
from transformers import VisionTextDualEncoderConfig

logger = logging.get_logger(__name__)


class SpeechWithEncoderDecoderConfig(PretrainedConfig):


    model_type = "speech-text-dual-encoder-decoder"
    is_composition = True

    def __init__(self, projection_dim=512, logit_scale_init_value=2.6592, **kwargs):

        super().__init__(**kwargs)
        if "speech_encoder" not in kwargs or "encoder_decoder" not in kwargs:
            raise ValueError(
                f"A configuraton of type {self.model_type} cannot be instantiated because "
                f"not both `encoder` and `decoder` sub-configurations are passed, but only {kwargs}"
            )


        speech_encoder_config = kwargs.pop("speech_encoder")

        encoder_decoder_config = kwargs.pop("encoder_decoder")

        decoder_config = kwargs.pop("decoder")

        self.speech_encoder = speech_encoder_config
        self.encoder_decoder = encoder_decoder_config
        self.decoder = decoder_config
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        # self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
        self.is_encoder_decoder = True

    @classmethod
    def from_encoder_decoder_configs(
        cls,
        speech_encoder_config: PretrainedConfig,
        encoder_decoder_config: PretrainedConfig,
        decoder_config: PretrainedConfig,
        **kwargs
    ) -> PretrainedConfig:

        logger.info("Setting `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config")

        # import pdb
        # pdb.set_trace()

        return cls(speech_encoder=speech_encoder_config,
                   encoder_decoder = encoder_decoder_config,
                   decoder = decoder_config,
                  **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["speech_encoder"] = self.speech_encoder.to_dict()
        # output["text_encoder"] = self.text_encoder.to_dict()
        output["encoder_decoder"] = self.encoder_decoder.to_dict()
        output["decoder"] = self.decoder.to_dict()

        output["model_type"] = self.__class__.model_type
        return output




