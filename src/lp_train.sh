GENERAL_DOMAIN=amazon
DOMAIN=sports
PROCESSED_DIR=data_dir/$GENERAL_DOMAIN/$DOMAIN/also_bought
LOG_DIR=logs/$DOMAIN/also_bought
CHECKPOINT_DIR=ckpt/$DOMAIN/also_bought

LR="1e-5"
MODEL_TYPE=graphformer

MODEL_DIR=ckpt/$DOMAIN/mask_mlm/$MODEL_TYPE/1e-5/
# MODEL_DIR=ckpt/$DOMAIN/sci-pretrain/mask_mlm/$MODEL_TYPE/1e-5

echo "start training..."

# (Patton)
CUDA_VISIBLE_DEVICES=1 python -m OpenLP.driver.train  \
    --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/$LR  \
    --model_name_or_path $MODEL_DIR  \
    --model_type $MODEL_TYPE \
    --do_train  \
    --save_steps 100  \
    --eval_steps 100  \
    --logging_steps 100 \
    --train_path $PROCESSED_DIR/train.32.jsonl  \
    --eval_path $PROCESSED_DIR/val.32.jsonl  \
    --fp16  \
    --per_device_train_batch_size 128  \
    --per_device_eval_batch_size 256 \
    --learning_rate $LR  \
    --max_len 32  \
    --num_train_epochs 1000  \
    --logging_dir $LOG_DIR/$MODEL_TYPE/$LR  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --report_to tensorboard

# # (SciPatton)
# CUDA_VISIBLE_DEVICES=5 python -m OpenLP.driver.train  \
#     --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/$LR  \
#     --model_name_or_path $MODEL_DIR  \
#     --model_type $MODEL_TYPE \
#     --do_train  \
#     --save_steps 100  \
#     --eval_steps 100  \
#     --logging_steps 100 \
#     --train_path $PROCESSED_DIR/sci-pretrain/train.32.jsonl  \
#     --eval_path $PROCESSED_DIR/sci-pretrain/val.32.jsonl  \
#     --fp16  \
#     --per_device_train_batch_size 128  \
#     --per_device_eval_batch_size 256 \
#     --learning_rate $LR  \
#     --max_len 32  \
#     --num_train_epochs 1000  \
#     --logging_dir $LOG_DIR/$MODEL_TYPE/$LR  \
#     --evaluation_strategy steps \
#     --remove_unused_columns False \
#     --overwrite_output_dir True \
#     --report_to tensorboard
