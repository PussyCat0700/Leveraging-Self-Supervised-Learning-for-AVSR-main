export CUDA_VISIBLE_DEVICES=2

python eval.py \
--eval_lrs3_model_file /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/check/train-step_1191-wer_0.674.ckpt \
--modal VO \
--decode_type HYBRID_RESCORE \
--logname /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/decode_VO_big/decode_rescore_beam5_beta0.03.txt \
--beamWidth 5 \
--batch_size 48 \
--beta 0.03 \
--nbest 30 \


#nohup bash decode_VO_big/decode_rescore_beam5_beta0.03.sh &