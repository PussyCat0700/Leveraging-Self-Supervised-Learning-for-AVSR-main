export CUDA_VISIBLE_DEVICES=3

python eval.py \
--type kenlm \
--logname /mnt/lustre/sjtu/data2/alumni/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lrs23_4gram_2_1500_1_plus.txt \
--lmpath /mnt/lustre/sjtu/data2/alumni/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/LRS23_4gram.bin \
--lexicon /mnt/lustre/sjtu/data2/alumni/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lst/23plus.lst \
--beamthreshold 100.0 \
--wordscore 2 \
--beam 1500 \
--lmweight 1 \



# nohup bash grid_search1/sh/evaluate_4gram_2_1500_1_plus.sh > evaluate_4gram_2_1500_1_plus.log &