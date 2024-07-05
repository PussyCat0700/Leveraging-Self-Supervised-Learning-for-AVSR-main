export CUDA_VISIBLE_DEVICES=0

python eval.py \
--eval_lrs3_model_file /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/check/train-step_1191-wer_0.674.ckpt \
--modal VO \
--decode_type HYBRID_ORACLE \
--type kenlm \
--logname /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/decode_VO_result/hybird_oracle_48.txt \
--beam 500 \
--beamthreshold 25.0 \
--wordscore 1 \
--lmweight 1 \
--batch_size 48 \

# bash decode_VO_result/decode_hybrid_oracle_bs48.sh