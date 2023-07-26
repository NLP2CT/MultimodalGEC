ROOT=Your_path
SPEECH=$ROOT/speech-en 
SPEECH_DATA=$SPEECH/speech_data  ## speech data path

seed=333 
time=$(date "+%Y-%m-%d-%H-%M")

model_type=SpeechWithEncoderDecoderModel
echo $model_type
num_train_epochs=3

output_model=$SPEECH/model-result/model/dot-attention_bs-16x32_lr-0.0001-2a100-moe-mse
mkdir $SPEECH/model-result/result/dot-attention_bs-16x32_lr-0.0001-2a100-moe-mse
mkdir $output_model

for learning_rate in 0.0001;do
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_addr=localhost --master_port=54333 \
        src/gec_speech_moe_mse_en/main.py \
        --log_file $SPEECH/out/single_gpu_num_train_epochs_${num_train_epochs}_learning_rate_${learning_rate} \
        --learning_rate ${learning_rate} \
        --num_train_epochs ${num_train_epochs} \
        --speech_encoder_path "facebook/hubert-large-ls960-ft" \
        --text_encoder_path "t5-large" \
        --dataset_name multimodal \
        --speech_dir $SPEECH_DATA \
        --multimodal True \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 32 \
        --per_device_eval_batch_size 16 \
        --result_path_conll14 $SPEECH/model-result/result/dot-attention_bs-16x32_lr-0.0001-2a100-moe-mse/seed${seed}_lr${learning_rate}-conll14 \
        --result_path_bea19_test $SPEECH/model-result/result/dot-attention_bs-16x32_lr-0.0001-2a100-moe-mse/seed${seed}_lr${learning_rate}-bea19_test \
        --result_path_bea19_dev $SPEECH/model-result/result/dot-attention_bs-16x32_lr-0.0001-2a100-moe-mse/seed${seed}_lr${learning_rate}-bea19_dev \
        --result_path_conll13_dev $SPEECH/model-result/result/dot-attention_bs-16x32_lr-0.0001-2a100-moe-mse/seed${seed}_lr${learning_rate}-conll13 \
        --max_source_length 128 \
        --val_max_target_length 128 \
        --seed ${seed} \
        --model_type ${model_type} \
        --train_file "/your_json_data_path/cl8_json_differ_files/cl8_en_train.all.json" \
        --test_file_conll14 "/your_json_data_path/cl8_json_differ_files/coll14_test.json" \
        --test_file_bea19 "/your_json_data_path/cl8_json_differ_files/ABCN_bea19_test.json" \
        --eval_file_bea19_dev "/your_json_data_path/cl8_json_differ_files/ABCN_bea19_dev.json" \
        --eval_file_conll13_dev "/your_json_data_path/cl8_json_differ_files/coll13_dev.json" \
        --output_dir $output_model \
        --eval_epoch 1 \
        --eval_steps 2000 \
        --speech_learning_rate 1e-04 \
        --use_t5_model True \
        --source_prefix "translate English to English: " \
        --use_adafactor True \
        --preprocessing_num_workers 16
done



### Evaluation on the best model

for i in best; do

python "/tool/spacy_en_tok.py" \
    $SPEECH/model-result/result/dot-attention_bs-16x32_lr-0.0001-2a100-moe-mse/seed${seed}_lr${learning_rate}-conll14.$i.candidate \
    $SPEECH/model-result/result/dot-attention_bs-16x32_lr-0.0001-2a100-moe-mse/seed${seed}_lr${learning_rate}-conll14.$i.candidate.sptok

python "/tool/retokizier_en.py" \
    $SPEECH/model-result/result/dot-attention_bs-16x32_lr-0.0001-2a100-moe-mse/seed${seed}_lr${learning_rate}-conll14.$i.candidate.sptok \
    | tee $SPEECH/model-result/result/dot-attention_bs-16x32_lr-0.0001-2a100-moe-mse/seed${seed}_lr${learning_rate}-conll14.$i.candidate.spretok

python2 "/tool/scripts/m2scorer.py" -v \
    $SPEECH/model-result/result/dot-attention_bs-16x32_lr-0.0001-2a100-moe-mse/seed${seed}_lr${learning_rate}-conll14.$i.candidate.spretok  \
    "/your_path/official-2014.combined.m2" \
    | tee $SPEECH/model-result/result/dot-attention_bs-16x32_lr-0.0001-2a100-moe-mse/seed${seed}_lr${learning_rate}-conll14.$i.candidate.spretok.m2scores

done

