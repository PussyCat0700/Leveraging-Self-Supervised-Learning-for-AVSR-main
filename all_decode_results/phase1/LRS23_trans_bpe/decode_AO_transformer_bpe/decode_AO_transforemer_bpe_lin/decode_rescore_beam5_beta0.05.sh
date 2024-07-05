export CUDA_VISIBLE_DEVICES=7

python eval.py \
--decode_type HYBRID_RESCORE \
--logname /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/decode_AO_transformer_bpe/decode_rescore_beam5_beta0.05.txt \
--beamWidth 5 \
--batch_size 48 \
--beta 0.05 \

