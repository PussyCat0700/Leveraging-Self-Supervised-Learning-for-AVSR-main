export CUDA_VISIBLE_DEVICES=4

python eval.py \
--decode_type HYBRID_RESCORE \
--logname /home/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/decode_AO_transformer_bpe/decode_rescore_beam5_beta0.04.txt \
--beamWidth 5 \
--batch_size 48 \
--beta 0.04 \

