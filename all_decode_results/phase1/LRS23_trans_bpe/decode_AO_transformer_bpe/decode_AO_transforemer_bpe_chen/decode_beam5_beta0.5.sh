export CUDA_VISIBLE_DEVICES=1

python eval.py \
--decode_type HYBRID_LM \
--logname /mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/decode_AO_transforemer_bpe/decode_beam5_beta0.5.txt \
--batch_size 48 \
--beta 0.5 \
--beamWidth 5 \