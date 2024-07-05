
export CUDA_VISIBLE_DEVICES=2

python eval.py \
--eval_lrs3_model_file /data2/alumni/gryang/train-step_0144-cer_0.119.ckpt \
--modal VO \
--decode_type HYBRID_RESCORE \
--logname /data2/alumni/gryang/L2_train/vo_0.53.txt \
--beamWidth 20 \
--batch_size 64 \
--beta 0.53 \
--nbest 30 \


#nohup bash vo_0.53.sh &