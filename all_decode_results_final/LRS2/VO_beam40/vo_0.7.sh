
export CUDA_VISIBLE_DEVICES=7

python eval.py \
--eval_lrs3_model_file /home/gryang/train-step_0144-cer_0.119.ckpt \
--modal VO \
--decode_type HYBRID_RESCORE \
--logname /home/gryang/L2_train/VO_beam40/vo_0.7.txt \
--beamWidth 40 \
--batch_size 64 \
--beta 0.7 \
--nbest 30 \


#nohup bash VO_beam40/vo_0.7.sh &