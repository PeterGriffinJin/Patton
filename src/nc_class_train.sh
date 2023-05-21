GENERAL_DOMAIN=amazon
DOMAIN=sports
PROCESSED_DIR=data_dir/$GENERAL_DOMAIN/$DOMAIN/nc-coarse/8_8
LOG_DIR=logs/$DOMAIN/nc_class
CHECKPOINT_DIR=ckpt/$DOMAIN/nc_class

LR="1e-5"
MODEL_TYPE=graphformer

MODEL_DIR=ckpt/$DOMAIN/mask_mlm/$MODEL_TYPE/1e-5

echo "start training..."

# single GPU training
# (Patton)
CUDA_VISIBLE_DEVICES=0 python -m OpenLP.driver.train_class  \
    --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/$LR  \
    --model_name_or_path $MODEL_DIR  \
    --tokenizer_name 'bert-base-uncased' \
    --model_type $MODEL_TYPE \
    --do_train  \
    --save_steps 25  \
    --eval_steps 25  \
    --logging_steps 25 \
    --train_path $PROCESSED_DIR/train.jsonl  \
    --eval_path $PROCESSED_DIR/val.jsonl  \
    --class_num 16 \
    --fp16  \
    --per_device_train_batch_size 256  \
    --per_device_eval_batch_size 256 \
    --learning_rate $LR  \
    --max_len 32  \
    --num_train_epochs 500  \
    --logging_dir $LOG_DIR/$MODEL_TYPE/$LR  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --report_to tensorboard \
    --seed 42


# # (SciPatton)
# CUDA_VISIBLE_DEVICES=0 python -m OpenLP.driver.train_class  \
#     --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/$LR  \
#     --model_name_or_path $MODEL_DIR  \
#     --tokenizer_name "allenai/scibert_scivocab_uncased" \
#     --model_type $MODEL_TYPE \
#     --do_train  \
#     --save_steps 25  \
#     --eval_steps 25  \
#     --logging_steps 25 \
#     --train_path $PROCESSED_DIR/sci-pretrain/train.jsonl  \
#     --eval_path $PROCESSED_DIR/sci-pretrain/val.jsonl  \
#     --class_num 16 \
#     --fp16  \
#     --per_device_train_batch_size 256  \
#     --per_device_eval_batch_size 256 \
#     --learning_rate $LR  \
#     --max_len 32  \
#     --num_train_epochs 500  \
#     --logging_dir $LOG_DIR/$MODEL_TYPE/$LR  \
#     --evaluation_strategy steps \
#     --remove_unused_columns False \
#     --overwrite_output_dir True \
#     --report_to tensorboard \
#     --seed 42
