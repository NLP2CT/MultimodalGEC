import pdb

from datasets import load_dataset
from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer, BertTokenizer, T5Tokenizer
from PIL import Image
import os
from torchvision import transforms
# import librosa
# import torchaudio
import soundfile as sf


class GECMultiModalBaseDataset(Dataset):
    def __init__(self, args, path, tokenizer,
                 fields=("ungrammatical_text","speech_path" ,"grammatical_text"),
                 symbols=None,
                 is_train=True,
                 feature_extractor=None):


        self.args = args
        self.path = path
        self.examples = self.read_jsonl()

        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

        
        self.use_t5 = args.use_t5_model

        self.fields = fields
        self.bo_tgt = symbols["BOTgt"]
        self.eo_tgt = symbols["EOTgt"]
        self.bo_src = symbols["BOSrc"]
        self.eo_src = symbols["EOSrc"]

        self.train=is_train

        if args.val_max_target_length is None:
            self.val_max_target_length = args.max_target_length
        else:
            self.val_max_target_length = args.val_max_target_length

        # self.image_dir = image_dir
        self.speech_dir = args.speech_dir
        speech_dir = args.speech_dir



        for i in range(len(self.examples)):
            if self.train:
                example = self.examples[i]
                src_txt = example[fields[0]].replace('\n', '')


                tgt_txt = example[fields[-1]].replace('\n', '')

                speech_path = example[fields[1]]
                speech_file_path = os.path.join(speech_dir, speech_path)
                # import librosa

                # array, sampling_rate = sf.read(speech_file_path)
                # array = array.T

                tgt_ids, tgt_mask = self.tokenize_tgt(tgt_txt)

                
                if args.use_t5_model == True:
                    src_txt = args.source_prefix + src_txt


                self.examples[i][fields[0]] = src_txt
                self.examples[i][fields[-1]] = tgt_txt
                self.examples[i]['speech_file_path'] = speech_file_path
                self.examples[i]['tgt_ids'] = tgt_ids
                self.examples[i]['tgt_mask'] = tgt_mask

               
            else:
                example = self.examples[i]
                src_txt = example[fields[0]].replace('\n', '')

                if args.use_t5_model == True:
                    src_txt = args.source_prefix + src_txt

                speech_path = example[fields[1]]
                speech_file_path = os.path.join(speech_dir, speech_path)


                tgt_txt = example[fields[-1]].replace('\n', '')
                tgt_ids, tgt_mask = self.tokenize_tgt(tgt_txt)

                self.examples[i][fields[0]] = src_txt
                self.examples[i][fields[-1]] = tgt_txt
                self.examples[i]['speech'] = speech_file_path
                self.examples[i]['tgt_ids'] = tgt_ids
                self.examples[i]['tgt_mask'] = tgt_mask


    def tokenize_tgt(self, tgt_txt):
        tgt_txt_subtokens = self.tokenizer.tokenize(tgt_txt)
        if len(tgt_txt_subtokens)>self.val_max_target_length-2:
            tgt_txt_subtokens = tgt_txt_subtokens[:self.val_max_target_length-3]

        if self.use_t5 == True:
            # src_ids = self.tokenizer(tgt_txt,truncation=True,padding=True,max_length=self.val_max_target_length,return_tensors="pt")

            tgt_txt_subtokens = tgt_txt_subtokens + [self.eo_tgt]



        else:
            tgt_txt_subtokens = [self.bo_tgt] + tgt_txt_subtokens + [self.eo_tgt]

        tgt_ids = self.tokenizer.convert_tokens_to_ids(tgt_txt_subtokens)
        tgt_mask = [1] * len(tgt_ids)

        return tgt_ids, tgt_mask

    def read_jsonl(self):
        datasets_from_file = []
        with open(self.path,'r') as f:
            for line in f.readlines():
                datasets_from_file.append(json.loads(line))
        return datasets_from_file

    def __len__(self):
        return len(self.examples)


class GECMultiModalDataset(GECMultiModalBaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        source_text = example[self.fields[0]]
        tgt_text = example[self.fields[-1]]
        speech_file_path = example['speech_file_path']
        tgt_ids = example['tgt_ids']
        tgt_mask = example['tgt_mask']
        
        # for englsih .mv format
        array, sampling_rate = sf.read(speech_file_path) 
        speech = array.T

        ### for german .mp3 format
        # array, sampling_rate = torchaudio.load(speech_file_path, format="mp3")  
        # speech = array.numpy()
        # speech = speech.mean(axis=0)

        sample = (source_text, tgt_text, speech, tgt_ids, tgt_mask, speech_file_path)

        return sample
