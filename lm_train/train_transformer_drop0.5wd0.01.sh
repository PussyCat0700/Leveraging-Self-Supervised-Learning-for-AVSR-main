export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_NAME="LRS23vocab_LibriLRS23_transformer_drop0.5wd0.01"

fairseq-train --task language_modeling \
  /data2/alumni/gryang/fairseq/LRS23vocab_LibriLRS23_wordpiece/data-bin/LibriLRS23_wordpiece_vocab4000 \
  --save-dir /data2/alumni/gryang/fairseq/LRS23vocab_LibriLRS23_wordpiece/drop0.5wd0.01/ \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.5 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --decoder-embed-dim 1024 \
  --decoder-layers 12 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 83968 --update-freq 16 \
  --fp16 \
  --distributed-world-size 4 \
  --max-update 50000 \
  --keep-interval-updates -1 \
  --save-interval 10 \
  --validate-interval 10 \
  --wandb-project language_model \

  

# 45056  39*2048=79872
# nohup bash LRS23vocab_LibriLRS23_wordpiece/train_transformer_drop0.5wd0.01.sh > LRS23vocab_LibriLRS23_wordpiece/train_transformer_drop0.5wd0.01.log &