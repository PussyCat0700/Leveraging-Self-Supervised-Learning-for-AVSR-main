export CUDA_VISIBLE_DEVICES=3

python eval.py \
--decode_type HYBRID_RESCORE \
--type kenlm \
--logname /mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/decode_AO_transforemer_bpe/decode_rescore_beam5_beta0.1_comparescore.txt \
--batch_size 48 \
--beta 0.1 \
--beamWidth 5 \

