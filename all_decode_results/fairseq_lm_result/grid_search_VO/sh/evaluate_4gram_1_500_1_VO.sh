export CUDA_VISIBLE_DEVICES=2

python eval_VO.py \
--type kenlm \
--logname /mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lrs23_4gram_1_500_1_VO.txt \
--lmpath /mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/LRS23_4gram.bin \
--lexicon /mnt/lustre/sjtu/home/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lst/LRS23.lst \
--beamthreshold 100.0 \
--wordscore 1 \
--beam 500 \
--lmweight 1 \
