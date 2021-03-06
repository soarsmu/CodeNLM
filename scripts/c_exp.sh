#!/bin/bash
subword-nmt learn-bpe -s 10000 < ./data/c-corpus/training > ./data/c-corpus/bigcode/vocab_10000

subword-nmt apply-bpe -c ./data/c-corpus/bigcode/vocab_10000 < ./data/c-corpus/training > ./data/c-corpus/bigcode/training_10000 &

subword-nmt apply-bpe -c ./data/c-corpus/bigcode/vocab_10000 < ./data/c-corpus/test > ./data/c-corpus/bigcode/test_10000 &

subword-nmt apply-bpe -c ./data/c-corpus/bigcode/vocab_10000 < ./data/c-corpus/validation > ./data/c-corpus/bigcode/validation_10000

# split
subword-nmt learn-bpe -s 10000 < ./data/c-corpus/split/encoding > ./data/c-corpus/split/vocab_10000

subword-nmt apply-bpe -c ./data/c-corpus/split/vocab_10000 < ./data/c-corpus/training > ./data/c-corpus/split/training_10000 &

subword-nmt apply-bpe -c ./data/c-corpus/split/vocab_10000 < ./data/c-corpus/test > ./data/c-corpus/split/test_10000 &

subword-nmt apply-bpe -c ./data/c-corpus/split/vocab_10000 < ./data/c-corpus/validation > ./data/c-corpus/split/validation_10000

# merge
subword-nmt learn-bpe -s 10000 < ./data/c-corpus/merge/encoding > ./data/c-corpus/merge/vocab_10000

subword-nmt apply-bpe -c ./data/c-corpus/merge/vocab_10000 < ./data/c-corpus/training > ./data/c-corpus/merge/training_10000 &

subword-nmt apply-bpe -c ./data/c-corpus/merge/vocab_10000 < ./data/c-corpus/test > ./data/c-corpus/merge/test_10000 &

subword-nmt apply-bpe -c ./data/c-corpus/merge/vocab_10000 < ./data/c-corpus/validation > ./data/c-corpus/merge/validation_10000

# Maximum training epochs
EPOCHS=39
# Initial learning rate
LR=0.1 # This is the default value. You can skip it if you don't want to change it.
# Training batch size
BATCH_SIZE=64 # This is also the default.
# RNN unroll timesteps for gradient calculation.
STEPS=200 # This also the default.
# 1 - Dropout probability
KEEP_PROB=0.5 # This is also the default.
# RNN hidden state size
STATE_DIMS=512 # This is also the default.
# Checkpoint and validation loss calculation frequency.
CHECKPOINT_EVERY=5000 # This is also the default.

# Train a small py model for 1 epoch.
CUDA_VISIBLE_DEVICES=6 python code_nlm.py --data_path data/c-corpus/split --train_dir data/c-corpus/split/model_10000 --train_filename training_10000 --validation_filename validation_10000 --gru True --hidden_size $STATE_DIMS --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS | tee c_logs/split_10000.log &

CUDA_VISIBLE_DEVICES=5 python code_nlm.py --data_path data/c-corpus/bigcode --train_dir data/c-corpus/bigcode/model_10000 --train_filename training_10000 --validation_filename validation_10000 --gru True --hidden_size $STATE_DIMS --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS | tee c_logs/bpe_10000.log &

CUDA_VISIBLE_DEVICES=4 python code_nlm.py --data_path data/c-corpus/merge --train_dir data/c-corpus/merge/model_10000 --train_filename training_10000 --validation_filename validation_10000 --gru True --hidden_size $STATE_DIMS --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS | tee c_logs/merge_10000.log 

CUDA_VISIBLE_DEVICES=0 python code_nlm.py --data_path data/c_corpus --train_dir data/c_corpus/model_10000 --train_filename training_split_10000 --validation_filename validation_split_10000 --gru True --hidden_size 512 --batch_size 64 --word_level_perplexity True --cross_entropy True --steps_per_checkpoint 5000 --max_epoch 39 | tee c_logs/merge_10000.log


