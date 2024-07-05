export CUDA_VISIBLE_DEVICES=1

python eval.py \
--logname /mnt/lustre/sjtu/data2/alumni/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lrs3_trans_origin.txt \
--lmpath /mnt/lustre/sjtu/data2/alumni/gry10/fairseq/language_model/checkpoints/transformer_LRS3/checkpoint_best.pt \
--lexicon /mnt/lustre/sjtu/data2/alumni/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lst/pretrain_trainval.lst \
--beam 500 \
--beamthreshold 25.0 \
--wordscore 1 \