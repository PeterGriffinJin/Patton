GENERAL_DOMAIN=MAG
DOMAIN=Economics
PROCESSED_DIR=data_dir/$GENERAL_DOMAIN/$DOMAIN/sci-pretrain
LOG_DIR=logs/$DOMAIN/sci-pretrain
CHECKPOINT_DIR=ckpt/$DOMAIN/sci-pretrain

LR="1e-5"
MODEL_TYPE=graphformer

export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "start training..."

python -m torch.distributed.launch --nproc_per_node=4 --master_port 19299 \
    -m OpenLP.driver.train_with_mlm  \
    --output_dir $CHECKPOINT_DIR/mask_mlm/$MODEL_TYPE/$LR  \
    --model_name_or_path "allenai/scibert_scivocab_uncased"  \
    --model_type $MODEL_TYPE \
    --do_train  \
    --save_steps 20000  \
    --eval_steps 10000  \
    --logging_steps 1000 \
    --train_path $PROCESSED_DIR/train.jsonl  \
    --eval_path $PROCESSED_DIR/val.jsonl  \
    --fp16  \
    --per_device_train_batch_size 128  \
    --per_device_eval_batch_size 256 \
    --learning_rate $LR  \
    --max_len 32  \
    --num_train_epochs 30  \
    --logging_dir $LOG_DIR/mask_mlm/$MODEL_TYPE/$LR  \
    --evaluation_strategy steps \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --report_to tensorboard \
    --mlm_loss True \
    --dataloader_num_workers 32
