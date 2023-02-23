export CUDA_VISIBLE_DEVICES=5

python eval.py \
--decode_type HYBRID_LM \
--logname /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/decode_AO_transformer_bpe/decode_beam10_beta0.05.txt \
--batch_size 48 \
--beta 0.05 \
--beamWidth 10 \