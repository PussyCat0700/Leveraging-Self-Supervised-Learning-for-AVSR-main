export CUDA_VISIBLE_DEVICES=3

python eval.py \
--type kenlm \
--logname /mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lrs3_4gram_2_1500_1_plus.txt \
--lmpath /mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/LRS3_4gram.bin \
--lexicon /mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lst/plus.lst \
--beamthreshold 100.0 \
--wordscore 2 \
--beam 1500 \
--lmweight 1 \
