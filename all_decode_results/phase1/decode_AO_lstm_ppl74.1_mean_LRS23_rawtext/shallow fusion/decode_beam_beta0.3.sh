export CUDA_VISIBLE_DEVICES=2

python eval.py \
--decode_type HYBRID_LM \
--type kenlm \
--logname /mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/decode_beam_beta0.3.txt \
--beam 500 \
--beamthreshold 25.0 \
--wordscore 1 \
--lmweight 1 \
--batch_size 48 \
--beta 0.3 \