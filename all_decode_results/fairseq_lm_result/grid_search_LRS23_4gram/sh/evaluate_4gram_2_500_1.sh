export CUDA_VISIBLE_DEVICES=0

python eval.py \
--type kenlm \
--logname /mnt/lustre/sjtu/data2/alumni/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lrs23_4gram_2_500_1.txt \
--lmpath /mnt/lustre/sjtu/data2/alumni/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/LRS23_4gram.bin \
--lexicon /mnt/lustre/sjtu/data2/alumni/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lst/LRS23.lst \
--beamthreshold 100.0 \
--wordscore 2 \
--beam 500 \
--lmweight 1 \
