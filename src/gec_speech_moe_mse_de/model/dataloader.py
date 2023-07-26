import pdb

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizer, AutoProcessor
# from model.dataset import GECMultiModalDataset
from .dataset import GECMultiModalDataset

class CollateFn:
    def __init__(self, args, tokenizer, feature_extractor, is_train=True):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

        self.max_source_length = args.max_source_length
        # self.max_target_length = args.max_target_length
        self.use_prompt = args.use_prompt
        self.train = is_train

    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data


    def __call__(self, data):
        if self.train:
            # (source_text, tgt_text, image, tgt_ids, tgt_mask)

            src_txt, tgt_txt, speech_batch, tgt_ids, tgt_mask, speech_file_path = zip(*data)

            speech_list = []
            for item in speech_batch:
                speech_list.append(item)
            import librosa

            speech_inputs = self.feature_extractor(
                    speech_list,
                    sampling_rate=self.feature_extractor.sampling_rate,
                    padding=True,
                    max_length=20000,
                    truncation=True,
                    return_tensors="pt"
                ).input_values

            src_txt_list = []
            for example in src_txt:
                src_txt_list.append(example)


            src_ids = self.tokenizer(src_txt_list,
                               truncation=True,
                               padding=True,
                               max_length=self.max_source_length,
                               return_tensors="pt")

            tgt_txt_list = []
            for example in tgt_txt:
                tgt_txt_list.append(example)

            tgt_ids_list = []
            for tgt_example in tgt_ids:
                tgt_ids_list.append(tgt_example)

            tgt_mask_list = []
            for tgt_example in tgt_mask:
                tgt_mask_list.append(tgt_example)

            lengths = [len(bs_x) for bs_x in tgt_ids_list]
            max_lengths = max(lengths)

            pad_tgt_ids = torch.LongTensor(len(tgt_ids),max_lengths).fill_(self.tokenizer.pad_token_id)
            pad_mask_ids = torch.LongTensor(len(tgt_ids),max_lengths).fill_(0)

            for i, (tgt_, mask_) in enumerate(zip(tgt_ids_list, tgt_mask_list)):
                length = lengths[i]
                pad_tgt_ids[i, :length] = torch.LongTensor(tgt_)
                pad_mask_ids[i, :length] = torch.LongTensor(mask_)


            if self.use_prompt:
                return src_ids, pad_tgt_ids, pad_mask_ids, speech_inputs, src_txt, tgt_txt, speech_file_path
            else:
                return src_ids, pad_tgt_ids, pad_mask_ids, speech_inputs, src_txt, tgt_txt, speech_file_path
        else:
            # src_txt = zip(*data)

            src_txt_list = []
            for example in data:
                src_txt_list.append(example)

            src_ids = self.tokenizer(src_txt_list,
                                     truncation=True,
                                     padding=True,
                                     max_length=self.max_source_length,
                                     return_tensors="pt")
            return src_ids



class MyDataLoader(DataLoader):
    def __init__(self, args, tokenizer, feature_extractor, dataset, sampler=None, shuffle=None,pin_memory=True):
        self.args = args
        self.shuffle = shuffle
        self.num_workers = args.preprocessing_num_workers
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

        self.dataset = dataset

        self.collate_fn = CollateFn(args, tokenizer=tokenizer,feature_extractor=feature_extractor, is_train=self.dataset.train)


        if self.dataset.train:
            self.batch_size = args.per_device_train_batch_size
        else:
            self.batch_size = args.per_device_eval_batch_size

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'sampler': sampler,
            'pin_memory': pin_memory
        }

        super().__init__(**self.init_kwargs)



