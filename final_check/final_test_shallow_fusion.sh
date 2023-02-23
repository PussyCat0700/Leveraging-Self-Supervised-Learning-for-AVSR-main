export CUDA_VISIBLE_DEVICES=3

python eval.py \
--eval_lrs3_model_file /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/check/train-step_0604-wer_0.054.ckpt \
--modal AO \
--decode_type HYBRID_SHALLOW_FUSION \
--logname /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/final_check_hybrid_shallow.txt \
--beamWidth 5 \
--batch_size 48 \
--beta 0.02 \
--nbest 30 \


#nohup bash decode_AO_drop0.5wd0.01/decode_rescore_beam5_beta0.07.sh &