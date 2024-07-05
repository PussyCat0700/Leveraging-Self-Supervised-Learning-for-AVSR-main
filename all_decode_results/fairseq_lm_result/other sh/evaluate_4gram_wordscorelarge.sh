export CUDA_VISIBLE_DEVICES=3

python eval.py \
--type kenlm \
--logname /mnt/lustre/sjtu/data2/alumni/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lrs3_4gram_wordscorelarge.txt \
--lmpath /mnt/lustre/sjtu/data2/alumni/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/LRS3_4gram.bin \
--lexicon /mnt/lustre/sjtu/data2/alumni/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lst/pretrain_trainval.lst \
--beam 1500 \
--beamthreshold 100.0 \
--wordscore 3 \