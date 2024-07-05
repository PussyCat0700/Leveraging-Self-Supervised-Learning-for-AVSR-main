export CUDA_VISIBLE_DEVICES=2

python eval.py \
--eval_lrs3_model_file /data2/alumni/gryang/L_store/check925/train-step_0088-wer_0.057.ckpt \
--modal AV \
--decode_type HYBRID_RESCORE \
--logname /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/decode_AV_drop0.5wd0.01/decode_rescore_beam5_beta0.08_nbest5.txt \
--beamWidth 5 \
--batch_size 48 \
--beta 0.08 \
--nbest 5 \


#nohup bash decode_AV_drop0.5wd0.01/decode_rescore_beam5_beta0.08_nbest5.sh &