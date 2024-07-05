export CUDA_VISIBLE_DEVICES=0

python eval.py \
--decode_type HYBRID_RESCORE \
--type kenlm \
--logname /mnt/lustre/sjtu/data2/alumni/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/decode_AO_transforemer_bpe/decode_rescore_beam5_beta0.02.txt \
--batch_size 48 \
--beta 0.02 \
--beamWidth 5 \

