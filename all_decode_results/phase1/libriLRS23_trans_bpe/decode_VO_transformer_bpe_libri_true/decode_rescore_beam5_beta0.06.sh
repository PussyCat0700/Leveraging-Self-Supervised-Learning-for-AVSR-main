export CUDA_VISIBLE_DEVICES=5

python eval.py \
--eval_lrs3_model_file /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/check/train-step_1191-wer_0.674.ckpt \
--modal VO \
--decode_type HYBRID_RESCORE \
--logname /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/decode_AO_tranformer_bpe_libri_true/decode_rescore_beam5_beta0.06.txt \
--beamWidth 5 \
--batch_size 48 \
--beta 0.06 \
--nbest 30 \


#nohup bash decode_VO_transformer_bpe_libri_true/decode_rescore_beam5_beta0.06.sh &