export CUDA_VISIBLE_DEVICES=6

python eval.py \
--decode_type HYBRID_RESCORE \
--logname /data2/alumni/gryang/Leveraging-Self-Supervised-Learning-for-AVSR-main_noneed/decode_AO_transformer_bpe/decode_rescore_beam10_beta0.05.txt \
--beamWidth 10 \
--batch_size 48 \
--beta 0.05 \

