export CUDA_VISIBLE_DEVICES=3

python eval.py \
--eval_lrs3_model_file /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/check/train-step_1191-wer_0.674.ckpt \
--modal VO \
--decode_type HYBRID_RESCORE \
--logname /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/beam20/vo_0.5.txt \
--beamWidth 20 \
--batch_size 48 \
--beta 0.5 \
--nbest 30 \


#nohup bash beam20/vo_0.5.sh &