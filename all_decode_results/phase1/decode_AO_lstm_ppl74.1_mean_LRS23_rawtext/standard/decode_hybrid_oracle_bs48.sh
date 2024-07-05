export CUDA_VISIBLE_DEVICES=2

python eval.py \
--decode_type HYBRID_ORACLE \
--type kenlm \
--logname /mnt/lustre/sjtu/data2/alumni/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/hybird_oracle_48.txt \
--beam 500 \
--beamthreshold 25.0 \
--wordscore 1 \
--lmweight 1 \
--batch_size 48 \
