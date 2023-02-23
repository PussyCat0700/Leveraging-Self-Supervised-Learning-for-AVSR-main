export CUDA_VISIBLE_DEVICES=5

python eval.py \
--eval_lrs3_model_file /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/check/train-step_1191-wer_0.674.ckpt \
--modal VO \
--decode_type HYBRID_RESCORE \
--type kenlm \
--logname /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main/decode_VO_result/decode_rescore_beta1.txt \
--beam 500 \
--beamthreshold 25.0 \
--wordscore 1 \
--lmweight 1 \
--batch_size 48 \
--beta 1 \

#nohup bash decode_VO_result/decode_rescore_beta1.sh > decode_rescore_beta0.6.log &