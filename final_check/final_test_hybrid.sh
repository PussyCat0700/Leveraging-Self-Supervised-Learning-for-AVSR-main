export CUDA_VISIBLE_DEVICES=2

python eval.py \
--eval_lrs3_model_file /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/check/train-step_0604-wer_0.054.ckpt \
--modal AO \
--decode_type HYBRID \
--logname /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/final_check_hybrid.txt \
--beamWidth 5 \
--batch_size 48 \
--beta 0.07 \
--nbest 30 \


#nohup bash decode_AO_drop0.5wd0.01/decode_rescore_beam5_beta0.07.sh &