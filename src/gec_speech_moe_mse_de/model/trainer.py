import math
import torch
import os
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    BertTokenizer,
)

from transformers import BertLMHeadModel
import codecs

import numpy as np
# from datasets import load_metric
from tqdm.auto import tqdm
import nltk
from .loss import LabelSmoother, LabelSmoother_wo_log_softmax
from rouge import Rouge
from datasets import load_metric
from transformers.optimization import Adafactor

def build_optimizer(args, model):
    if 'bart-' in args.model_name_or_path or 'bigbird' in args.model_name_or_path \
            or 't5-' in args.model_name_or_path or 'long' in args.model_name_or_path \
            or 'led' in args.model_name_or_path:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        if args.use_adafactor == True:
            optimizer = Adafactor(optimizer_grouped_parameters, scale_parameter=False, relative_step=False, warmup_init=False, lr=args.learning_rate)
        else:
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)


    return optimizer



class BaseTrainer(object):
    def __init__(self,  args, model, accelerator, logger, tokenizer=None, symbols=None, num_gpus=1):
        self.args = args
        self.model = model
        self.num_train_epochs = args.num_train_epochs
        self.report_to = args.report_to

        self.output_dir = args.output_dir
        self.with_tracking = args.with_tracking
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.num_beams = args.num_beams

        self.num_gpus = num_gpus


        if args.val_max_target_length is None:
            self.val_max_target_length = args.max_target_length
        else:
            self.val_max_target_length = args.val_max_target_length


        self.logger = logger
        self.eval_epoch = args.eval_epoch
        self.eval_steps = args.eval_steps

        self.tokenizer = tokenizer


        self.pad_to_max_length = args.pad_to_max_length
        self.ignore_pad_token_for_loss = args.ignore_pad_token_for_loss



        self.test_metric = load_metric("rouge")

        self.optimizer = build_optimizer(args, model)
        self.accelerator = accelerator


        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None


        #####bertabs

        # self.tokenizer = tokenizer
        self.symbols = symbols
        self.alpha = args.alpha
        self.min_target_length = args.min_target_length

        self.start_token = self.tokenizer.vocab[symbols['BOTgt']]
        self.end_token = self.tokenizer.vocab[symbols['EOTgt']]
        self.block_trigram = args.block_trigram
        #####bertabs




    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        return preds, labels

    def train(self, train_dataloader, eval_dataloader, test_dataloader):

        self.train_autoregressive(train_dataloader, eval_dataloader, test_dataloader)



    def train_autoregressive(self, train_dataloader, eval_dataloader, test_dataloader):



        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        else:
            self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)


        self.lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )


        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.gradient_accumulation_steps)
        max_train_steps = self.num_train_epochs * num_update_steps_per_epoch
        total_batch_size = self.per_device_train_batch_size * self.accelerator.num_processes * self.gradient_accumulation_steps
        progress_bar = tqdm(range(max_train_steps), disable=not self.accelerator.is_local_main_process)

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(train_dataloader)}")
        self.logger.info(f"  Num Epochs = {self.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.per_device_train_batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {max_train_steps}")
        self.logger.info(f"  num_beams = {self.num_beams}")
        self.logger.info(f"  generate_max_target_length = {self.val_max_target_length}")
        self.logger.info(f"  generate_min_target_length = {self.min_target_length}")
        self.logger.info(f"  learning_rate = {self.args.learning_rate}")
        self.logger.info(f"  use_adafactor = {self.args.use_adafactor}")
        self.logger.info(f"  use_t5_model = {self.args.use_t5_model}")
        self.logger.info(f"  t5_source_prefix = {self.args.source_prefix}")


        # Only show the progress bar once on each machine.
        completed_steps = 0
        starting_epoch = 0
        gradient_accumulation_loss = 0

        for epoch in range(starting_epoch, self.num_train_epochs):
            self.model.train()
            if self.with_tracking:
                total_loss = 0

            if self.num_gpus>1:
                train_dataloader.sampler.set_epoch(epoch)

            for step, batch in enumerate(train_dataloader):

                if self.args.multimodal:
                    if self.args.use_prompt:
                        src_ids, pad_tgt_ids, pad_mask_ids, speech_ids, src_txt, tgt_txt, speech_file_path, clss = batch

                        clss = clss.cuda()
                    else:
                        clss = None
                        src_ids, pad_tgt_ids, pad_mask_ids, speech_ids, src_txt, tgt_txt, speech_file_path = batch

                    speech_ids = speech_ids.cuda()

                else:
                    speech_ids = None
                    src_ids, pad_tgt_ids, pad_mask_ids, src_txt, tgt_txt = batch

                src_ids['labels'] = pad_tgt_ids.cuda()
                input_ids = src_ids['input_ids'].cuda()
                attention_mask = src_ids['attention_mask'].cuda()
                labels = pad_tgt_ids.cuda()

                if self.args.multimodal:
                    outputs = self.model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         labels=labels,
                                         speech_ids=speech_ids,
                                         clss=clss)
                else:
                    outputs = self.model(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         labels=labels,
                                         )

                loss = outputs.loss
                gradient_accumulation_loss = gradient_accumulation_loss + loss.detach().float()
                # We keep track of the loss at each epoch
                if self.with_tracking:
                    total_loss += loss.detach().float()

                loss = loss / self.gradient_accumulation_steps

                loss.backward()

                if step % self.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    progress_bar.update(1)
                    progress_bar.set_description("(Epoch {}) LOSS:{:.4f}".format(step, gradient_accumulation_loss))
                    completed_steps += 1
                    gradient_accumulation_loss = 0

            #         # print("----------------------------------------------------------------------------")
            #         # print(str(completed_steps))
            #         # print(speech_file_path)
            #         # print(src_txt) 
            #         # print(tgt_txt)
            #         # print("----------------------------------------------------------------------------")

                            
                    # if completed_steps % self.eval_steps == 0 and completed_steps != 0:

                    #     self.test_autoregressive(eval_dataloader, completed_steps)


            if (epoch + 1) % self.eval_epoch == 0:
                self.test_autoregressive(test_dataloader, 1, epoch)
                self.test_autoregressive(eval_dataloader, 2, epoch)
                # # # save model
                # self.save_model(epoch)

    def test_autoregressive(self, test_dataloader, tag, epoch=0):


        test_dataloader = self.accelerator.prepare(test_dataloader)

        gen_kwargs = {
            "max_length": self.val_max_target_length,
            "num_beams": self.num_beams,
            # "min_length": self.min_target_length,
            # "no_repeat_ngram_size": 3
            # "eos_token_id": self.end_token
        }
        samples_seen = 0
        predictions = []
        references = []

        progress_bar_test = tqdm(range(len(test_dataloader)))

        self.model.eval()
        for step, batch in enumerate(test_dataloader):
            with torch.no_grad():

                if self.args.multimodal:
                    if self.args.use_prompt:
                        #src_ids, pad_tgt_ids, pad_mask_ids, speech_ids, src_txt, tgt_txt, clss = batch
                        src_ids, pad_tgt_ids, pad_mask_ids, speech_ids, src_txt, tgt_txt, speech_file_path, clss = batch

                        # src_ids, pad_tgt_ids, pad_mask_ids, image_batch, src_txt, tgt_txt, clss

                        clss = clss.cuda()
                    else:
                        clss = None
                        #src_ids, pad_tgt_ids, pad_mask_ids, speech_ids, src_txt, tgt_txt = batch
                        src_ids, pad_tgt_ids, pad_mask_ids, speech_ids, src_txt, tgt_txt, speech_file_path = batch

                    speech_ids = speech_ids.cuda()

                else:
                    speech_ids = None
                    src_ids, pad_tgt_ids, pad_mask_ids, src_txt, tgt_txt = batch

                input_ids = src_ids['input_ids'].cuda()
                attention_mask = src_ids['attention_mask'].cuda()

                # from transformers.generation_logits_process import NoRepeatNGramLogitsProcessor, \
                #     MinLengthLogitsProcessor, ForcedBOSTokenLogitsProcessor, ForcedEOSTokenLogitsProcessor
                #
                # logit_process_1 = NoRepeatNGramLogitsProcessor(ngram_size=3)
                # logit_process_2 = MinLengthLogitsProcessor(min_length=self.min_target_length,eos_token_id=self.tokenizer.eos_token_id)
                # logit_process_3 = ForcedBOSTokenLogitsProcessor(bos_token_id=self.tokenizer.bos_token_id)
                # logit_process_4 = ForcedEOSTokenLogitsProcessor(max_length=self.val_max_target_length,eos_token_id=self.tokenizer.eos_token_id)
                # logits_processor = [logit_process_1,logit_process_2,logit_process_3,logit_process_4]

                if self.args.multimodal:
                    generated_tokens = self.accelerator.unwrap_model(self.model).generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        speech_ids=speech_ids,
                        clss=clss,
                        **gen_kwargs,
                    )
                else:
                    generated_tokens = self.accelerator.unwrap_model(self.model).generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **gen_kwargs,
                    )
                # self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=image_batch)

                generated_tokens = self.accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id
                )
                labels = pad_tgt_ids

                if not self.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = self.accelerator.pad_across_processes(labels, dim=1,
                                                                   pad_index=self.tokenizer.pad_token_id)

                # generated_tokens, labels = self.accelerator.gather((generated_tokens, labels))
                # generated_tokens = generated_tokens.cpu().numpy()
                # labels = labels.cpu().numpy()

                generated_tokens = self.accelerator.gather(generated_tokens).cpu().numpy()
                labels = self.accelerator.gather(labels).cpu().numpy()

                if self.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

                ##self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True,clean_up_tokenization_spaces=False)

                decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
                # If we are in a multiprocess environment, the last batch has duplicates
                if self.accelerator.num_processes > 1:
                    if step == len(test_dataloader) - 1:
                        decoded_preds = decoded_preds[: len(test_dataloader.dataset) - samples_seen]
                        decoded_labels = decoded_labels[: len(test_dataloader.dataset) - samples_seen]
                    else:
                        samples_seen += len(decoded_labels)


                # print(step)
                self.test_metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )

                predictions = predictions+decoded_preds
                references = references+decoded_labels

                progress_bar_test.update(1)

        if tag == 1:
            gold_path = self.args.result_path_de_test + '.%d.gold' % epoch
            can_path = self.args.result_path_de_test + '.%d.candidate' % epoch
            self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
            self.can_out_file = codecs.open(can_path, 'w', 'utf-8')

            process_predictions = []
            process_references = []


            for i in range(len(predictions)):
                pred, gold = predictions[i], references[i]


                # pred = pred.replace('[unused1]','').replace('[unused2]','').replace('[unused0]','').replace('\n','')
                # pred = covert_string_to_evaluate_formate(pred)
                #
                # gold = gold.replace('[unused1]','').replace('[unused2]','').replace('[unused0]','').replace('\n','')
                # gold = covert_string_to_evaluate_formate(gold)



                if pred == '':
                    pred = 'Nothing Generated'

                process_predictions.append(pred)
                process_references.append(gold)

                self.can_out_file.write(pred + '\n')
                self.gold_out_file.write(gold + '\n')


            self.can_out_file.flush()
            self.gold_out_file.flush()

            self.can_out_file.close()
            self.gold_out_file.close()

            rouge = Rouge()
            rouge_score = rouge.get_scores(process_predictions, process_references, avg=True)
            r1 = rouge_score['rouge-1']['f'] * 100
            r2 = rouge_score['rouge-2']['f'] * 100
            rl = rouge_score['rouge-l']['f'] * 100
            result = dict()
            result['rouge-1'] = r1
            result['rouge-2'] = r2
            result['rouge-l'] = rl
            self.logger.info(f"  Rouge result = {result}")

        if tag == 2:
            gold_path = self.args.result_path_de_dev + '.%d.gold' % epoch
            can_path = self.args.result_path_de_dev + '.%d.candidate' % epoch
            self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
            self.can_out_file = codecs.open(can_path, 'w', 'utf-8')

            process_predictions = []
            process_references = []


            for i in range(len(predictions)):
                pred, gold = predictions[i], references[i]


                # pred = pred.replace('[unused1]','').replace('[unused2]','').replace('[unused0]','').replace('\n','')
                # pred = covert_string_to_evaluate_formate(pred)
                #
                # gold = gold.replace('[unused1]','').replace('[unused2]','').replace('[unused0]','').replace('\n','')
                # gold = covert_string_to_evaluate_formate(gold)



                if pred == '':
                    pred = 'Nothing Generated'

                process_predictions.append(pred)
                process_references.append(gold)

                self.can_out_file.write(pred + '\n')
                self.gold_out_file.write(gold + '\n')


            self.can_out_file.flush()
            self.gold_out_file.flush()

            self.can_out_file.close()
            self.gold_out_file.close()

            rouge = Rouge()
            rouge_score = rouge.get_scores(process_predictions, process_references, avg=True)
            r1 = rouge_score['rouge-1']['f'] * 100
            r2 = rouge_score['rouge-2']['f'] * 100
            rl = rouge_score['rouge-l']['f'] * 100
            result = dict()
            result['rouge-1'] = r1
            result['rouge-2'] = r2
            result['rouge-l'] = rl
            self.logger.info(f"  Rouge result = {result}")


    
    def save_model(self, epoch):
        # model_path = self.args.summary_field+'_2_epoch_'+str(epoch)+'.pt'
        model_path = os.path.join(self.output_dir, "moe_mse_cl8_de_checkpoint_epoch_{}.pt".format(epoch))

        if self.num_gpus > 1:
            torch.save(self.model.module, model_path)
        else:
            torch.save(self.model, model_path)



