
import os
import gc
#from keras import backend as K
import torch


from transformers import (
    AutoConfig,
    AutoTokenizer,
    SchedulerType,
    BartForConditionalGeneration,
    BertTokenizer,
    T5Tokenizer,
    AutoModelWithLMHead,
    AutoModelForSeq2SeqLM,
    EncoderDecoderModel,
    AutoFeatureExtractor
)
import torch
import argparse
from model.trainer import BaseTrainer
from model.dataloader import MyDataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
import logging
from model.symbols import symbols_dict
from model.dataset import GECMultiModalDataset
from torch.utils.data.distributed import DistributedSampler
from model.speech_with_encoder_decoder.modeling_speech_with_encoder_decoder import SpeechWithEncoderDecoderModel
import random
import numpy as np



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


tokenizer_path = "/home/derekfw/yc074909/T5-code/model_room/pretrain_bart_checkpoint/"


def get_params():
    # parse parameters
    parser = argparse.ArgumentParser(description="Cross-domain NER")
    parser.add_argument("--model_name_or_path", type=str, default='/home/derekfw/yc074909/tao/model/bart-', )

    parser.add_argument("--dump_path", type=str, default="experiments5", help="Experiment saved root path")
    parser.add_argument("--exp_id", type=str, default="1", help="Experiment id")


    parser.add_argument("--decoder_model_name_or_path", type=str, default=None )


    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--dev_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )

    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )


    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--use_t5_model",
        type=bool,
        default=False,
        help="If set True, a prefix should to add before every source text (only for T5 models).",
    )
    parser.add_argument(
        "--use_adafactor",
        type=bool,
        default=False,
        help="If set True, used for T5 models.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=0,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )



    parser.add_argument(
        "--min_target_length",
        type=int,
        default=5,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )


    parser.add_argument(
        "--max_target_length",
        type=int,
        default=40,
        help=(
            "The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=60,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )

    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    ) 
    parser.add_argument(
        "--eval_epoch",
        type=int,
        default=2,
        help="Evaluation for each epoch",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Evaluation for each epoch",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--speech_learning_rate",
        type=float,
        default=3e-4,
        help="Initial learning rate for speech encodder to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=200, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )

    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    parser.add_argument(
        "--log_file",
        type=str,
        default='./log/mimic-cxr.log',
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)


    parser.add_argument("--summary_field", default='出院情况', type=str)


    parser.add_argument("--model_type", default='SpeechWithEncoderDecoderModel', type=str, help=('SpeechWithEncoderDecoderModel'))


    parser.add_argument("--do_train", default=True, type = bool)

    # type = bool,
    # default = True,

    parser.add_argument("--multimodal", default=False, type = bool)

    parser.add_argument("--use_prompt", default=False, type = bool)



    parser.add_argument("--speech_encoder_path", default='', type=str)
    parser.add_argument("--text_encoder_path", default='', type=str)
    parser.add_argument("--modal_joint_model", default='', type=str)
    parser.add_argument("--decoder_path", default='', type=str)

    parser.add_argument("--image_dir", type=str, default='/mntnfs/diis_data3/chenqian/workspace/R2Gen-main/data/mimic_cxr')
    parser.add_argument("--speech_dir", type=str, default='/Users/hjp/phd/multimodal_data/speech')


    ##bertabs
    parser.add_argument("--max_pos", default=512, type=int)
    parser.add_argument("--enc_hidden_size", default=512, type=int)
    parser.add_argument("--enc_ff_size", default=512, type=int)
    parser.add_argument("--enc_dropout", default=0.2, type=float)
    parser.add_argument("--enc_layers", default=6, type=int)
    parser.add_argument("--enc_heads", default=8, type=int)
    parser.add_argument("--dec_dropout", default=0.2, type=float)
    parser.add_argument("--dec_layers", default=6, type=int)
    parser.add_argument("--dec_hidden_size", default=768, type=int)
    parser.add_argument("--dec_heads", default=8, type=int)
    parser.add_argument("--dec_ff_size", default=2048, type=int)
    parser.add_argument("--vocab_size", default=21128, type=int)
    parser.add_argument("--alpha", default=1, type=int)
    parser.add_argument("--result_path", default='mimic-cxr')

    parser.add_argument("--result_path_de_test", default='mimic-cxr')
    parser.add_argument("--result_path_de_dev", default='mimic-cxr')

    parser.add_argument("--lr_encoder", default=0.2, type=float)
    parser.add_argument("--lr_decoder", default=0.2, type=float)

    parser.add_argument("--warmup_steps_encoder", default=0.2, type=int)
    parser.add_argument("--warmup_steps_decoder", default=0.2, type=int)

    parser.add_argument(
        "--block_trigram",
        action="store_true",
    )
    parser.add_argument("--label_smoothing_factor", default=0.1, type=float)


    params = parser.parse_args()

    return params



def main():
    ####test_for_bart
    args = get_params()


    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://',rank=args.local_rank)
    logger = get_logger(__name__)
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    )

    log_file = args.log_file
    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        # logger.addHandler(file_handler)
    console = logging.StreamHandler()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[file_handler, console]
    )

    set_seed(args.seed)
    tokenizer = None
    feature_extractor = None

    if args.multimodal:
        model_type = args.model_type

        # modal_joint_model = args.modal_joint_model

        speech_encoder_path = args.speech_encoder_path
        text_encoder_path = args.text_encoder_path
        decoder_path = args.decoder_path
        if model_type == 'SpeechWithEncoderDecoderModel':
            model = SpeechWithEncoderDecoderModel.from_encoder_decoder_pretrained(
                speech_encoder_pretrained_model_name_or_path=speech_encoder_path,
                text_encoder_pretrained_model_name_or_path=text_encoder_path)

            # ### multimodal
            # model2 = torch.load(args.modal_joint_model,map_location='cuda:0')
            # model.load_state_dict(model2.state_dict())
            # model2 = model2.cpu()
            # gc.collect()
            # del model2
            # torch.cuda.empty_cache()

            #tokenizer = T5Tokenizer.from_pretrained(text_encoder_path)
            tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
            feature_extractor = AutoFeatureExtractor.from_pretrained(speech_encoder_path)
            if args.use_t5_model == True:
                symbols = symbols_dict['symbols_t5']
            else:
                symbols = symbols_dict['symbols_bart']

    num_gpus=1
    model = model.cuda()

    if args.local_rank != -1:
        # model.to(device)
        num_gpus = torch.cuda.device_count()
        print('num_gpus',num_gpus)
    if num_gpus > 1:
        print('use {} gpus!'.format(num_gpus))
        
        # Trainer = BaseTrainer(args,
        #                     model,
        #                     accelerator,
        #                     logger,
        #                     symbols = symbols,
        #                     num_gpus = num_gpus,
        #                     tokenizer = tokenizer)

        model = torch.nn.parallel.DistributedDataParallel(model.cuda(),
                                                          device_ids=[args.local_rank],

                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    Trainer = BaseTrainer(args,
                          model,
                          accelerator,
                          logger,
                          symbols = symbols,
                          num_gpus = num_gpus,
                          tokenizer = tokenizer)

    
#    for name, para in model.named_parameters():
#        print("{}: {}".format(name, para))

#    print("lm head para ------------------------------------------------------------------------------------")
#    for p in model.lm_head.parameters():
 #       print(p)

    # print("speech-encode para ------------------------------------------------------------------------------------")
    #         # # speech-encoder
    # for p in model.encoder.speech_model.parameters():
    #     p.requires_grad = False
    #     print(p)
            


    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)
    if not tokenizer:
        if 'bart-large-chinese' in args.model_name_or_path or \
                'bart-base-chinese' in args.model_name_or_path:
            tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)
        else:
            #tokenizer = T5Tokenizer.from_pretrained(text_encoder_path)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)


    if args.dataset_name == 'multimodal':

        if args.use_prompt:
            fields = ("ungrammatical_text","speech_path" ,"grammatical_text")
            logging.info('load train data ... ')

            train_dataset = GECMultiModalDataset(args, path=args.train_file,
                                                      tokenizer=tokenizer,
                                                      fields=fields,
                                                      symbols=symbols,
                                                      is_train=True)
            logging.info('finish loading train data ... ')

            logging.info('load eval data ... ')
            eval_dataset = GECMultiModalDataset(args, path=args.dev_file,
                                                     tokenizer=tokenizer,
                                                     fields=fields,
                                                     symbols=symbols,
                                                     is_train=True)

            logging.info('finish loading eval data ... ')

            logging.info('load test data ... ')
            test_dataset = GECMultiModalDataset(args, path=args.test_file,
                                                     tokenizer=tokenizer,
                                                     fields=fields,
                                                     symbols=symbols,
                                                     is_train=True)

            logging.info('finish loading test data ... ')



        else:
            fields = ("ungrammatical_text","speech_path" ,"grammatical_text")


            logging.info('load train data ... ')

            train_dataset = GECMultiModalDataset(args, path=args.train_file,
                                           tokenizer = tokenizer,
                                           fields=fields,
                                           symbols=symbols,
                                           is_train=True)

            logging.info('finish loading train data ... ')

            logging.info('load eval data ... ')
            eval_dataset = GECMultiModalDataset(args, path=args.dev_file,
                                           tokenizer=tokenizer,
                                           fields=fields,
                                           symbols=symbols,
                                           is_train=True)

            logging.info('finish loading eval data ... ')

            logging.info('load test data ... ')
            test_dataset = GECMultiModalDataset(args, path=args.dev_file,
                                           tokenizer=tokenizer,
                                           fields=fields,
                                           symbols=symbols,
                                           is_train=True)

            logging.info('finish loading test data ... ')





    if num_gpus>1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

        train_dataloader = MyDataLoader(args,
                                    tokenizer=tokenizer,
                                    feature_extractor=feature_extractor,
                                    dataset=train_dataset,
                                    sampler=train_sampler,
                                    pin_memory=True)
    else:
        train_dataloader = MyDataLoader(args,
                                        tokenizer=tokenizer,
                                        feature_extractor=feature_extractor,
                                        dataset=train_dataset,
                                        shuffle=True)


    eval_dataloader = MyDataLoader(args,
                                   tokenizer=tokenizer,
                                   feature_extractor=feature_extractor,
                                   shuffle=False,
                                   dataset=eval_dataset)

    test_dataloader = MyDataLoader(args,
                                   tokenizer=tokenizer,
                                   feature_extractor=feature_extractor,
                                   shuffle=False,
                                   dataset=test_dataset)


    Trainer.train(train_dataloader, eval_dataloader, test_dataloader)

          

if __name__ == '__main__':

    main()

    # generate()
