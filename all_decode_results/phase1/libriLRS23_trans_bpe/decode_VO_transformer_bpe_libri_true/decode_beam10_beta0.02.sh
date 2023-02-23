export CUDA_VISIBLE_DEVICES=2

python eval.py \
--eval_lrs3_model_file /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/check/train-step_1191-wer_0.674.ckpt \
--modal VO \
--decode_type HYBRID_LM \
--logname /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/decode_VO_transformer_bpe_libri_true/decode_beam10_beta0.02.txt \
--batch_size 48 \
--beta 0.02 \
--beamWidth 10 \
--nbest 30 \

# nohup bash decode_VO_transformer_bpe_libri_true/decode_beam10_beta0.02.sh &