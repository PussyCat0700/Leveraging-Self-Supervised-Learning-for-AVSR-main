export CUDA_VISIBLE_DEVICES=0

python eval.py \
--eval_lrs3_model_file /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/check/train-step_1191-wer_0.674.ckpt \
--modal VO \
--decode_type HYBRID_RESCORE \
--logname /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/beam40_LRS3/vo_0.8.txt \
--beamWidth 40 \
--batch_size 48 \
--beta 0.8 \
--nbest 30 \


#nohup bash beam40_LRS3/vo_0.8.sh &