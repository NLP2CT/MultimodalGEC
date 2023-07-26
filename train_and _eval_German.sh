ROOT=Your-path
SPEECH=$ROOT/cl8-speech-german

seed=222
time=$(date "+%Y-%m-%d-%H-%M")

model_type=SpeechWithEncoderDecoderModel
echo $model_type
num_train_epochs=21

mkdir $SPEECH/result/Geman-2a100-8x32-lr0.0002-mt5-ft-cl8-german

for learning_rate in 0.0002;do

TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_addr=localhost --master_port=54321 \
        src/gec_speech_moe_mse_de/main.py \
        --log_file $SPEECH/out/single_gpu_num_train_epochs_${num_train_epochs}_learning_rate_${learning_rate} \
        --learning_rate ${learning_rate} \
        --num_train_epochs ${num_train_epochs} \
        --speech_encoder_path "facebook/wav2vec2-xls-r-300m" \
        --text_encoder_path "google/mt5-large" \
        --dataset_name multimodal \
        --speech_dir $SPEECH/data \
        --multimodal True \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 32 \
        --per_device_eval_batch_size 8 \
        --result_path_de_dev /$SPEECH/result/Geman-2a100-8x32-lr0.0002-mt5-ft-cl8-german/seed${seed}_german_2a100_lr_dev_${learning_rate} \
        --result_path_de_test /$SPEECH/result/Geman-2a100-8x32-lr0.0002-mt5-ft-cl8-german/seed${seed}_german_2a100_lr_test_${learning_rate} \
        --max_source_length 128 \
        --val_max_target_length 128 \
        --seed ${seed} \
        --model_type ${model_type} \
        --train_file "/your-path/cl8-speech-german/data/jsons/train-cl8.de.json" \
        --dev_file "/your-path/cl8-speech-german/data/jsons/dev.de.json" \
        --test_file "/your-path/cl8-speech-german/data/jsons/test.de.json" \
        --output_dir /your-path/cl8-speech-german/model/pre_cl8-de/ \
        --eval_epoch 3 \
        --speech_learning_rate 1e-04 \
        --use_t5_model True \
        --source_prefix "translate German to German: " \
        --use_adafactor True \
        --preprocessing_num_workers 16
done


### Evaluation on the best model

for i in best; do

python "/tool/spacy_de_tok.py" \
    /$SPEECH/result/Geman-2a100-8x32-lr0.0002-mt5-ft-cl8-german/seed${seed}_german_2a100_lr_test_${learning_rate}.$i.candidate \
    /$SPEECH/result/Geman-2a100-8x32-lr0.0002-mt5-ft-cl8-german/seed${seed}_german_2a100_lr_test_${learning_rate}.$i.candidate.sptok

python2 "/tool/scripts/m2scorer.py" -v \
     /$SPEECH/result/Geman-2a100-8x32-lr0.0002-mt5-ft-cl8-german/seed${seed}_german_2a100_lr_test_${learning_rate}.$i.candidate.sptok  \
    "/your_path/fm-test-de.m2" \
    | tee /$SPEECH/result/Geman-2a100-8x32-lr0.0002-mt5-ft-cl8-german/seed${seed}_german_2a100_lr_test_${learning_rate}.$i.candidate.sptok.m2scores

done
