export CUDA_VISIBLE_DEVICES=1

python eval.py \
--type fairseqlm \
--logname /mnt/lustre/sjtu/data2/alumni/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lrs3_trans_fast_2_1500_1.txt \
--lmpath /mnt/lustre/sjtu/data2/alumni/gry10/fairseq/language_model/LMcheckpoints/transformer_LRS3_second/checkpoint_best.pt \
--lexicon /mnt/lustre/sjtu/data2/alumni/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lst/pretrain_trainval.lst \
--beamthreshold 25.0 \
--wordscore 2 \
--beam 1500 \
--lmweight 1 \


#nohup bash grid_search_trans_fast/sh/eval_trans_fast_2_1500_1.sh > eval_trans_fast_2_1500_1.log &