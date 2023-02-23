export CUDA_VISIBLE_DEVICES=5

python eval.py \
--eval_lrs3_model_file /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/check/train-step_1191-wer_0.674.ckpt \
--modal VO \
--decode_type HYBRID_LM \
--type kenlm \
--logname /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/decode_beam_beta0.3.txt \
--beam 500 \
--beamthreshold 25.0 \
--wordscore 1 \
--lmweight 1 \
--batch_size 48 \
--beta 0.3 \