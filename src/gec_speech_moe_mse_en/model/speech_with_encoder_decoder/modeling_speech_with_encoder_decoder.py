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
""" PyTorch BART model."""
import copy
import math
import random
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN

from transformers import (
    ViTModel,
    # BartForConditionalGeneration,
    AutoModel,
    CLIPVisionModel,
    CLIPVisionConfig,
    PretrainedConfig,
    AutoConfig,
    AutoModelForSeq2SeqLM)

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import is_torch_fx_proxy


from transformers.utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

#from model.bart.modeling_bart import BartForConditionalGeneration
from transformers import BartForConditionalGeneration
from .configuration_speech_with_encoder_decoder import SpeechWithEncoderDecoderConfig
from .moe import MoE


from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.modeling_outputs import ModelOutput

#import torch
#import deepspeed
#from deepspeed.moe.layer import MoE

# from configuration_speech_with_encoder_decoder import SpeechWithEncoderDecoderConfig



logger = logging.get_logger(__name__)


BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-large",
    # see all BART models at https://huggingface.co/models?filter=bart
]


# t5
def shift_tokens_right(input_ids, pad_token_id: int, decoder_start_token_id: int):
        # decoder_start_token_id = self.config.decoder_start_token_id
        # pad_token_id = self.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
            " See T5 docs for more information"
        )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

# # bart
# def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
#     """
#     Shift input ids one token to the right.
#     """
#     shifted_input_ids = input_ids.new_zeros(input_ids.shape)
#     shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
#     shifted_input_ids[:, 0] = decoder_start_token_id

#     if pad_token_id is None:
#         raise ValueError("self.model.config.pad_token_id has to be defined.")
#     # replace possible -100 values in labels by `pad_token_id`
#     shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

#     return shifted_input_ids


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(float("-inf")))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


@dataclass
class MultiModalLMOutput(ModelOutput):
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    loss_contrastive: Optional[torch.FloatTensor] = None
    attention_mask: Optional=None
    last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out # bs X output_size


class JointSpeechTextEncoder(PreTrainedModel):
    def __init__(self, config, speech_encoder, text_encoder, embed_tokens=None):
        # super(JointSpeechTextEncoder,self).__init__()
        super().__init__(config)

        self.speech_model = speech_encoder
        self.text_model = text_encoder

        self.speech_embed_dim = config.speech_encoder.hidden_size

        if config.encoder_decoder.model_type == 'bart' or config.encoder_decoder.model_type == 't5':
            self.text_embed_dim = config.encoder_decoder.d_model
        else:
            self.text_embed_dim = config.encoder_decoder.hidden_size

        self.speech_projection = nn.Linear(self.speech_embed_dim, self.text_embed_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.text_embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * self.config.logit_scale_init_value)
        self.num_expert = 6
        self.k = 3
        # self.margin = 0.1

        self.speech_MoE = MoE(input_size=self.speech_embed_dim, output_size=self.text_embed_dim, num_experts=self.num_expert, hidden_size=self.text_embed_dim, k=self.k, noisy_gating=True)
        # self.cosembedloss = nn.CosineEmbeddingLoss(margin=self.margin)
        self.mseloss = nn.MSELoss()

         

    def dot_attention(self, query, key, value, mask=None):

        attention_weight = query @ key.permute(0, 2, 1)

        src_batch = key.shape[0]
        src_len = key.shape[1]
        tgt_len = query.shape[1]

        if mask != None:
            mask = mask.unsqueeze(1).expand(src_batch, tgt_len, src_len)
            scores = attention_weight.masked_fill(mask, -1e18)
        else:
            scores = attention_weight
        attention_weight = torch.softmax(scores, dim=-1)
        attention_output = attention_weight @ value

        return attention_output

    def forward(
        self,
        input_ids=None,
        clss=None,
        speech_ids=None,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_loss=None,
        token_type_ids=None
    ):

        return_dict = return_dict if return_dict is not None else self.config.return_dict

        speech_outputs = self.speech_model(
            input_values=speech_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        text_encoder_hidden = text_outputs[0]
        speech_encoder_hidden = speech_outputs[0]

        if text_encoder_hidden.shape[-1] != speech_encoder_hidden.shape[-1]:
            speech_encoder_hidden = self.speech_projection(speech_encoder_hidden)
            # text_encoder_hidden = self.text_projection(text_encoder_hidden)

       

        encoder_hidden_states = self.dot_attention(text_encoder_hidden, speech_encoder_hidden, speech_encoder_hidden)
        encoder_hidden_states = encoder_hidden_states + text_encoder_hidden

        text_vec_mean = torch.mean(text_encoder_hidden, dim=1) # bs seq_lenth hidden_dim -> bs 1 hidden_dim
        att_feats_mean = torch.mean(speech_encoder_hidden, dim=1)  # bs seq_lenth hidden_dim -> bs 1 hidden_dim
        att_feats_mean, gate_loss = self.speech_MoE(att_feats_mean) # bs 1 hidden_dim
        # target_vec_mean = torch.mean(text_vec_mean, dim=1)

        hcl_loss = self.mseloss(text_vec_mean, att_feats_mean)


        return MultiModalLMOutput(
            last_hidden_state=encoder_hidden_states,
            loss_contrastive=hcl_loss 
            #last_hidden_state=text_encoder_hidden,
        )

class SpeechWithEncoderDecoderModel(PreTrainedModel):
    config_class = SpeechWithEncoderDecoderConfig

    def __init__(self,
                 config: Optional[PretrainedConfig] = None,
                 speech_encoder: Optional[PreTrainedModel] = None,
                 encoder_decoder: Optional[PreTrainedModel] = None,
                 ):

        if config is None and (speech_encoder is None or encoder_decoder is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = SpeechWithEncoderDecoderConfig.from_encoder_decoder_configs\
                (speech_encoder_config=speech_encoder.speech_model.config,
                 encoder_decoder_config=encoder_decoder.get_encoder().config,
                 decoder_config=encoder_decoder.get_decoder().config,)
        else:

            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        # import pdb
        # pdb.set_trace()

        # config.pad_token_id = config.encoder_decoder.pad_token_id
        # config.eos_token_id = config.encoder_decoder.eos_token_id
        # config.decoder_start_token_id = config.encoder_decoder.decoder_start_token_id
        # config.tie_word_embeddings = True
        config.decoder_start_token_id = config.encoder_decoder.pad_token_id
        config.pad_token_id = config.encoder_decoder.pad_token_id
        config.vocab_size = config.encoder_decoder.vocab_size
        config.eos_token_id = config.encoder_decoder.eos_token_id
        # config.decoder_start_token_id = config.encoder_decoder.decoder_start_token_id
        # config.pad_token_id = config.encoder_decoder.pad_token_id
        # config.vocab_size = config.encoder_decoder.vocab_size
        # config.eos_token_id = config.encoder_decoder.eos_token_id
        # config.forced_bos_token_id = config.encoder_decoder.forced_bos_token_id
        # config.forced_eos_token_id = config.encoder_decoder.forced_eos_token_id


        super().__init__(config)


        self.speech_encoder = speech_encoder
        #

        self.encoder = JointSpeechTextEncoder(config,
                                              speech_encoder = speech_encoder,
                                              text_encoder = encoder_decoder.get_encoder())
        self.decoder = encoder_decoder.get_decoder()

        # ###
        self.config.decoder = self.decoder.config
        self.lm_head = encoder_decoder.get_output_embeddings()

        # import pdb
        # pdb.set_trace()



    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @classmethod
    def from_encoder_decoder_pretrained(
            cls,
            speech_encoder_pretrained_model_name_or_path: str = None,
            text_encoder_pretrained_model_name_or_path: str = None,
            *model_args,
            **kwargs
    ) -> PreTrainedModel:

        kwargs_encoder = {
            argument[len("encoder_"):]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        # import pdb
        # pdb.set_trace()
        if encoder is None:
            if speech_encoder_pretrained_model_name_or_path is None or \
                    text_encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                speech_encoder_config, kwargs_speech_encoder = AutoConfig.from_pretrained(
                    speech_encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
                )

                if speech_encoder_config.is_decoder is True or speech_encoder_config.add_cross_attention is True:
                    logger.info(
                        f"Initializing {speech_encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    speech_encoder_config.is_decoder = False
                    speech_encoder_config.add_cross_attention = False

                text_encoder_config, kwargs_text_encoder = AutoConfig.from_pretrained(
                    text_encoder_pretrained_model_name_or_path, **kwargs_encoder, return_unused_kwargs=True
                )
                if text_encoder_config.is_encoder_decoder is True:
                    # encoder_decoder = AutoModel.from_pretrained(text_encoder_pretrained_model_name_or_path)

                    encoder_decoder = AutoModelForSeq2SeqLM.from_pretrained(text_encoder_pretrained_model_name_or_path)
                    # encoder_decoder =  BartForConditionalGeneration.from_pretrained(text_encoder_pretrained_model_name_or_path)

                    speech_encoder = AutoModel.from_pretrained(speech_encoder_pretrained_model_name_or_path)
                    # speech_encoder = AutoModel.from_pretrained(speech_encoder_pretrained_model_name_or_path)



        # instantiate config with corresponding kwargs
        config = SpeechWithEncoderDecoderConfig.from_encoder_decoder_configs \
            (speech_encoder_config=speech_encoder_config,
             encoder_decoder_config=encoder_decoder.get_encoder().config,
             decoder_config=encoder_decoder.get_decoder().config,
             **kwargs)

        # make sure input & output embeddings is not tied
        # config.tie_word_embeddings = False

        return cls(speech_encoder=speech_encoder, encoder_decoder=encoder_decoder, config=config)



    def forward(
        self,
        input_ids=None,
        encoder_attention_mask=None,
        encoder_outputs=None,
        clss=None,
        speech_ids=None,
        decoder_input_ids=None,
        decoder_inputs_embeds=None,
        decoder_attention_mask=None,
        decoder_head_mask=None,
        attention_mask=None,
        return_loss=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
        return_dict=None,
        use_cache=None,
        past_key_values=None,
        cross_attn_head_mask=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.encoder_decoder.use_return_dict
        # use_cache = use_cache if use_cache is not None else self.config.encoder_decoder.use_cache
        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}
        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.encoder_decoder.pad_token_id, self.config.encoder_decoder.decoder_start_token_id
                    #labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        else:
            use_cache = use_cache if use_cache is not None else self.config.encoder_decoder.use_cache

        if encoder_outputs is None:
            if speech_ids is None:
                raise ValueError("You have to specify speech_ids")

            encoder_outputs = self.encoder(
                input_ids=input_ids,
                speech_ids=speech_ids,
                attention_mask=attention_mask,
                return_loss=return_loss,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                clss=clss,
                **kwargs_encoder)

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):

            # encoder_outputs = BaseModelOutput(*encoder_outputs)
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs['last_hidden_state'],
                hidden_states=encoder_outputs['hidden_states'] if len(encoder_outputs) > 2 else None,
                attentions=encoder_outputs['attentions'] if len(encoder_outputs) > 3 else None,
            )
            # # encoder_outputs = BaseModelOutput(*encoder_outputs)
            # encoder_outputs = BaseModelOutput(
            #     last_hidden_state=encoder_outputs[0],
            #     hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
            #     attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            # )


        # encoder_hidden_states = encoder_outputs[0]

        # if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
        #     decoder_input_ids = shift_tokens_right(
        #         labels, self.config.encoder_decoder.pad_token_id, self.config.encoder_decoder.decoder_start_token_id
        #     )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs['last_hidden_state'],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs_decoder,
        )


        sequence_output = decoder_outputs[0]
        if self.decoder.config.model_type=="t5":
            sequence_output = sequence_output * (self.decoder.config.d_model**-0.5)


        logits = self.lm_head(sequence_output)


        # loss = None
        # if labels is not None:
        #     # logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

        loss = None
        if labels is not None:
            # logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

            loss_contrastive = encoder_outputs['loss_contrastive']

            loss = loss + 0.1 * loss_contrastive



        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs


        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.decoder.pad_token_id, self.config.decoder.decoder_start_token_id)
        #return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)


    def prepare_inputs_for_generation(
        self,
        decoder_input_ids=None,
        speech_ids=None,
        clss=None,
        past=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        **kwargs
    ):

        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]


        input_dict = {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }



        return input_dict

    def _reorder_cache(self, past, beam_idx):
        # apply decoder cache reordering here
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


# if __name__ == '__main__':
#     from model.dataset import GECMultiModalDataset
#     from model.dataloader import MyDataLoader
#     import argparse
#     from transformers import AutoTokenizer, AutoFeatureExtractor
#     parser = argparse.ArgumentParser(description="Cross-domain NER")
#     parser.add_argument("--model_name_or_path", type=str,
#                         default='/mntnfs/med_data2/jinpeng/workspace/bert-base-uncased', )
#
#     parser.add_argument(
#         "--preprocessing_num_workers",
#         type=int,
#         default=0,
#         help="The number of processes to use for the preprocessing.",
#     )
#
#     parser.add_argument(
#         "--per_device_train_batch_size",
#         type=int,
#         default=16,
#         help="Batch size (per device) for the training dataloader.",
#     )
#
#     parser.add_argument(
#         "--val_max_target_length",
#         type=int,
#         default=128,
#         help="Batch size (per device) for the training dataloader.",
#     )
#     parser.add_argument(
#         "--speech_dir",
#         type=str,
#         default='/Users/hjp/phd/multimodal_data',
#     )
#     parser.add_argument(
#         "--per_device_eval_batch_size",
#         type=int,
#         default=16,
#         help="Batch size (per device) for the evaluation dataloader.",
#     )
#     parser.add_argument(
#         "--max_source_length",
#         type=int,
#         default=512,
#         help="Batch size (per device) for the evaluation dataloader.",
#     )
#
#     parser.add_argument("--use_prompt", default=False, type=bool)
#
#     parser.add_argument(
#         "--test_file", type=str, default='/Users/hjp/phd/multimodal_data/speech/json_files/coll14_test.json'
#     )
#
#     parser.add_argument(
#         "--ignore_pad_token_for_loss",
#         type=bool,
#         default=True,
#         help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
#     )
#     params = parser.parse_args()
#     args = params
#
#     vision_encoder_path = '/Users/hjp/My_project/Pre-trained Models/wav2vec2-base-960h'
#     text_encoder_path='/Users/hjp/My_project/Pre-trained Models/bart_large_ft'
#
#     tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
#     feature_extractor = AutoFeatureExtractor.from_pretrained(vision_encoder_path)
#     fields = ("ungrammatical_text","speech_path" ,"grammatical_text")
#
#     symbols_bart = {
#         "BOTgt": "<s>",
#         "EOTgt": "</s>",
#         "PAD": "<pad>",
#         "BOSrc": "<s>",
#         "EOSrc": "</s>",
#     }
#
#
#
#     # eval_dataset = GECMultiModalDataset(args, path=args.test_file,
#     #                                     tokenizer=tokenizer,
#     #                                     fields=fields,
#     #                                     symbols=symbols_bart,
#     #                                     is_train=True)
#     #
#     # eval_dataloader = MyDataLoader(args,
#     #                                tokenizer=tokenizer,
#     #                                feature_extractor=feature_extractor,
#     #                                shuffle=False,
#     #                                dataset=eval_dataset)
#
#
#     text = ['Keeping the Secret of Genetic Testing']
#
#     inputs = tokenizer(text, return_tensors="pt", padding=True)
#     model = SpeechWithEncoderDecoderModel.from_encoder_decoder_pretrained(
#         speech_encoder_pretrained_model_name_or_path=vision_encoder_path,
#         text_encoder_pretrained_model_name_or_path=text_encoder_path)
#
#     # import pdb
#     # pdb.set_trace()
#     gen_kwargs = {
#         "max_length": 128,
#         "num_beams": 5,
#         "min_length": 1,
#         # "eos_token_id": self.end_token
#     }
#     from transformers.generation_logits_process import NoRepeatNGramLogitsProcessor, \
#         MinLengthLogitsProcessor, ForcedBOSTokenLogitsProcessor, ForcedEOSTokenLogitsProcessor
#
#     logit_process_1 = NoRepeatNGramLogitsProcessor(ngram_size=3)
#     logit_process_2 = MinLengthLogitsProcessor(min_length=1,
#                                                eos_token_id=tokenizer.eos_token_id)
#     logit_process_3 = ForcedBOSTokenLogitsProcessor(bos_token_id=tokenizer.bos_token_id)
#     logit_process_4 = ForcedEOSTokenLogitsProcessor(max_length=128,
#                                                     eos_token_id=tokenizer.eos_token_id)
#     # logits_processor = [logit_process_1, logit_process_2, logit_process_3, logit_process_4]
#
#     sequences = model.generate(input_ids=inputs.input_ids,
#                    attention_mask=inputs.attention_mask,
#                     **gen_kwargs,)
#
#     captions = tokenizer.batch_decode(sequences, skip_special_tokens=True)
#
#
#     print(captions)