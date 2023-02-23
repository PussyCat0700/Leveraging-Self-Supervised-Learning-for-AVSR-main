export CUDA_VISIBLE_DEVICES=3

python eval.py \
--eval_lrs3_model_file /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/check/train-step_0604-wer_0.054.ckpt \
--modal AO \
--decode_type HYBRID_RESCORE \
--logname /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/decode_AO_drop0.5wd0.01/decode_rescore_beam5_beta0.09.txt \
--beamWidth 5 \
--batch_size 48 \
--beta 0.09 \
--nbest 30 \


#nohup bash decode_AO_drop0.5wd0.01/decode_rescore_beam5_beta0.09.sh &