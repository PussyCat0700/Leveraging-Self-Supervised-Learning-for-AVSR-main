export CUDA_VISIBLE_DEVICES=1

python eval.py \
--decode_type HYBRID_RESCORE \
--type kenlm \
--logname /mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/decode_AO_transforemer_bpe/decode_rescore_beam5_beta0.04.txt \
--batch_size 48 \
--beta 0.04 \
--beamWidth 5 \

