export CUDA_VISIBLE_DEVICES=0

python eval.py \
--decode_type HYBRID_LM \
--type kenlm \
--logname /mnt/lustre/sjtu/data2/alumni/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/decode_AO_result/decode_beam_beta0.txt \
--beam 500 \
--beamthreshold 25.0 \
--wordscore 1 \
--beamWidth 5 \
--batch_size 48 \
--beta 0 \