export CUDA_VISIBLE_DEVICES=7

python eval.py \
--eval_lrs3_model_file /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/check/train-step_1191-wer_0.674.ckpt \
--modal VO \
--decode_type HYBRID_RESCORE \
--logname /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/beam20/vo_0.2.txt \
--beamWidth 20 \
--batch_size 48 \
--beta 0.2 \
--nbest 30 \


#nohup bash beam20/vo_0.2.sh &