export CUDA_VISIBLE_DEVICES=2

python eval.py \
--type kenlm \
--logname /mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lrs23_4gram_2_500_2.txt \
--lmpath /mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/LRS23_4gram.bin \
--lexicon /mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lst/LRS23.lst \
--beamthreshold 100.0 \
--wordscore 2 \
--beam 500 \
--lmweight 2 \


# nohup bash grid_search1/sh/evaluate_4gram_2_500_2.sh > evaluate_4gram_2_500_2.log &