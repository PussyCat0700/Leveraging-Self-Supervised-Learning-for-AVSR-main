export CUDA_VISIBLE_DEVICES=1

python eval.py \
--decode_type HYBRID_LM \
--logname /mnt/lustre/sjtu/data2/alumni/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/decode_AO_transforemer_bpe/decode_beam5_beta0.3.txt \
--batch_size 48 \
--beta 0.3 \
--beamWidth 5 \