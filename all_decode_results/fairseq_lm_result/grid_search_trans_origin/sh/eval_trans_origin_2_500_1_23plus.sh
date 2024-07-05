export CUDA_VISIBLE_DEVICES=3

python eval.py \
--type fairseqlm \
--logname /mnt/lustre/sjtu/data2/alumni/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lrs3_trans_origin_2_500_1_23plus.txt \
--lmpath /mnt/lustre/sjtu/data2/alumni/gry10/fairseq/language_model/checkpoints/transformer_LRS3/checkpoint_best.pt \
--lexicon /mnt/lustre/sjtu/data2/alumni/gry10/Leveraging-Self-Supervised-Learning-for-AVSR-main/lst/23plus.lst \
--beamthreshold 25.0 \
--wordscore 2 \
--beam 500 \
--lmweight 1 \


#nohup bash grid_search_trans_origin/sh/eval_trans_origin_2_500_1_23plus.sh > eval_trans_origin_2_500_1_23plus.log &