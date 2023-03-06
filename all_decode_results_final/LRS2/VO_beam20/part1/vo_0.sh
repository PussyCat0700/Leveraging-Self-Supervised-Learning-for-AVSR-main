export CUDA_VISIBLE_DEVICES=2

python eval.py \
--eval_lrs3_model_file /home/gryang/train-step_0144-cer_0.119.ckpt \
--modal VO \
--decode_type HYBRID_RESCORE \
--logname /home/gryang/L2_train/vo_0.txt \
--beamWidth 20 \
--batch_size 64 \
--beta 0 \
--nbest 30 \


#nohup bash vo_0.sh &