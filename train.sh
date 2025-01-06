echo " Running Training EXP"

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5 --tag social-stgcnn-nfl2025 --use_lrschd --num_epochs 250 && echo "NFL Launched."
