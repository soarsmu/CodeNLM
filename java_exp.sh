# #!/bin/bash
# subword-nmt learn-bpe -s 2000 < ./data/java-corpus/bigcode/java_training_pre > ./data/java-corpus/bigcode/encoding_new.bpe_2000 &

# subword-nmt learn-bpe -s 5000 < ./data/java-corpus/bigcode/java_training_pre > ./data/java-corpus/bigcode/encoding_new.bpe_5000 &

# subword-nmt learn-bpe -s 10000 < ./data/java-corpus/bigcode/java_training_pre > ./data/java-corpus/bigcode/encoding_new.bpe_10000

# subword-nmt apply-bpe -c ./data/java-corpus/bigcode/encoding_new.bpe_2000 < ./data/java-corpus/bigcode/java_training_pre > ./data/java-corpus/bigcode/training_2000 &

# subword-nmt apply-bpe -c ./data/java-corpus/bigcode/encoding_new.bpe_5000 < ./data/java-corpus/bigcode/java_training_pre > ./data/java-corpus/bigcode/training_5000 &

# subword-nmt apply-bpe -c ./data/java-corpus/bigcode/encoding_new.bpe_10000 < ./data/java-corpus/bigcode/java_training_pre > ./data/java-corpus/bigcode/training_10000 &

# subword-nmt apply-bpe -c ./data/java-corpus/bigcode/encoding_new.bpe_2000 < ./data/java-corpus/bigcode/java_test_pre > ./data/java-corpus/bigcode/test_2000 &

# subword-nmt apply-bpe -c ./data/java-corpus/bigcode/encoding_new.bpe_5000 < ./data/java-corpus/bigcode/java_test_pre > ./data/java-corpus/bigcode/test_5000 &

# subword-nmt apply-bpe -c ./data/java-corpus/bigcode/encoding_new.bpe_10000 < ./data/java-corpus/bigcode/java_test_pre > ./data/java-corpus/bigcode/test_10000 &

# subword-nmt apply-bpe -c ./data/java-corpus/bigcode/encoding_new.bpe_2000 < ./data/java-corpus/bigcode/java_validation_pre > ./data/java-corpus/bigcode/validation_2000 &

# subword-nmt apply-bpe -c ./data/java-corpus/bigcode/encoding_new.bpe_5000 < ./data/java-corpus/bigcode/java_validation_pre > ./data/java-corpus/bigcode/validation_5000 &

# subword-nmt apply-bpe -c ./data/java-corpus/bigcode/encoding_new.bpe_10000 < ./data/java-corpus/bigcode/java_validation_pre > ./data/java-corpus/bigcode/validation_10000

# # split
# subword-nmt learn-bpe -s 2000 < ./data/java-corpus/split/java_training_split > ./data/java-corpus/split/encoding.bpe_2000 &

# subword-nmt learn-bpe -s 5000 < ./data/java-corpus/split/java_training_split > ./data/java-corpus/split/encoding.bpe_5000 &

# subword-nmt learn-bpe -s 10000 < ./data/java-corpus/split/java_training_split > ./data/java-corpus/split/encoding.bpe_10000

# subword-nmt apply-bpe -c ./data/java-corpus/split/encoding.bpe_2000 < ./data/java-corpus/bigcode/java_training_pre > ./data/java-corpus/split/training_2000 &

# subword-nmt apply-bpe -c ./data/java-corpus/split/encoding.bpe_5000 < ./data/java-corpus/bigcode/java_training_pre > ./data/java-corpus/split/training_5000 &

# subword-nmt apply-bpe -c ./data/java-corpus/split/encoding.bpe_10000 < ./data/java-corpus/bigcode/java_training_pre > ./data/java-corpus/split/training_10000 &

# subword-nmt apply-bpe -c ./data/java-corpus/split/encoding.bpe_2000 < ./data/java-corpus/bigcode/java_test_pre > ./data/java-corpus/split/test_2000 &

# subword-nmt apply-bpe -c ./data/java-corpus/split/encoding.bpe_5000 < ./data/java-corpus/bigcode/java_test_pre > ./data/java-corpus/split/test_5000 &

# subword-nmt apply-bpe -c ./data/java-corpus/split/encoding.bpe_10000 < ./data/java-corpus/bigcode/java_test_pre > ./data/java-corpus/split/test_10000 &

# subword-nmt apply-bpe -c ./data/java-corpus/split/encoding.bpe_2000 < ./data/java-corpus/bigcode/java_validation_pre > ./data/java-corpus/split/validation_2000 &

# subword-nmt apply-bpe -c ./data/java-corpus/split/encoding.bpe_5000 < ./data/java-corpus/bigcode/java_validation_pre > ./data/java-corpus/split/validation_5000 &

# subword-nmt apply-bpe -c ./data/java-corpus/split/encoding.bpe_10000 < ./data/java-corpus/bigcode/java_validation_pre > ./data/java-corpus/split/validation_10000

# # merge
# subword-nmt learn-bpe -s 2000 < ./data/java-corpus/merge/merge > ./data/java-corpus/merge/encoding_new.merge_2000 &

# subword-nmt learn-bpe -s 5000 < ./data/java-corpus/merge/merge > ./data/java-corpus/merge/encoding_new.merge_5000 &

# subword-nmt learn-bpe -s 10000 < ./data/java-corpus/merge/merge > ./data/java-corpus/merge/encoding_new.merge_10000

# subword-nmt apply-bpe -c ./data/java-corpus/merge/encoding_new.merge_2000 < ./data/java-corpus/bigcode/java_training_pre > ./data/java-corpus/merge/training_2000 &

# subword-nmt apply-bpe -c ./data/java-corpus/merge/encoding_new.merge_5000 < ./data/java-corpus/bigcode/java_training_pre > ./data/java-corpus/merge/training_5000 &

# subword-nmt apply-bpe -c ./data/java-corpus/merge/encoding_new.merge_10000 < ./data/java-corpus/bigcode/java_training_pre > ./data/java-corpus/merge/training_10000 &

# subword-nmt apply-bpe -c ./data/java-corpus/merge/encoding_new.merge_2000 < ./data/java-corpus/bigcode/java_test_pre > ./data/java-corpus/merge/test_2000 &

# subword-nmt apply-bpe -c ./data/java-corpus/merge/encoding_new.merge_5000 < ./data/java-corpus/bigcode/java_test_pre > ./data/java-corpus/merge/test_5000 &

# subword-nmt apply-bpe -c ./data/java-corpus/merge/encoding_new.merge_10000 < ./data/java-corpus/bigcode/java_test_pre > ./data/java-corpus/merge/test_10000 &

# subword-nmt apply-bpe -c ./data/java-corpus/merge/encoding_new.merge_2000 < ./data/java-corpus/bigcode/java_validation_pre > ./data/java-corpus/merge/validation_2000 &

# subword-nmt apply-bpe -c ./data/java-corpus/merge/encoding_new.merge_5000 < ./data/java-corpus/bigcode/java_validation_pre > ./data/java-corpus/merge/validation_5000 &

# subword-nmt apply-bpe -c ./data/java-corpus/merge/encoding_new.merge_10000 < ./data/java-corpus/bigcode/java_validation_pre > ./data/java-corpus/merge/validation_10000

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

# Train a small java model for 1 epoch.
CUDA_VISIBLE_DEVICES=0 python code_nlm.py --data_path data/java-corpus/split --train_dir data/java-corpus/split/model_2000 --train_filename training_2000 --validation_filename validation_2000 --hidden_size $STATE_DIMS --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS | tee java_logs/split_2000.log &

CUDA_VISIBLE_DEVICES=1 python code_nlm.py --data_path data/java-corpus/split --train_dir data/java-corpus/split/model_5000 --train_filename training_5000 --validation_filename validation_5000 --hidden_size $STATE_DIMS --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS | tee java_logs/split_5000.log &

CUDA_VISIBLE_DEVICES=2 python code_nlm.py --data_path data/java-corpus/split --train_dir data/java-corpus/split/model_10000 --train_filename training_10000 --validation_filename validation_10000 --hidden_size $STATE_DIMS --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS | tee java_logs/split_10000.log &

CUDA_VISIBLE_DEVICES=2 python code_nlm.py --data_path data/java-corpus/bigcode --train_dir data/java-corpus/bigcode/model_2000 --train_filename training_2000 --validation_filename validation_2000 --hidden_size $STATE_DIMS --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS | tee java_logs/bpe_2000.log &

CUDA_VISIBLE_DEVICES=1 python code_nlm.py --data_path data/java-corpus/bigcode --train_dir data/java-corpus/bigcode/model_5000 --train_filename training_5000 --validation_filename validation_5000 --hidden_size $STATE_DIMS --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS | tee java_logs/bpe_5000.log &

CUDA_VISIBLE_DEVICES=0 python code_nlm.py --data_path data/java-corpus/bigcode --train_dir data/java-corpus/bigcode/model_10000 --train_filename training_10000 --validation_filename validation_10000 --hidden_size $STATE_DIMS --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS | tee java_logs/bpe_10000.log &

CUDA_VISIBLE_DEVICES=1 python code_nlm.py --data_path data/java-corpus/merge --train_dir data/java-corpus/merge/model_2000 --train_filename training_2000 --validation_filename validation_2000 --hidden_size $STATE_DIMS --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS | tee java_logs/merge_2000.log &

CUDA_VISIBLE_DEVICES=2 python code_nlm.py --data_path data/java-corpus/merge --train_dir data/java-corpus/merge/model_5000 --train_filename training_5000 --validation_filename validation_5000 --hidden_size $STATE_DIMS --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS | tee java_logs/merge_5000.log &

CUDA_VISIBLE_DEVICES=0 python code_nlm.py --data_path data/java-corpus/merge --train_dir data/java-corpus/merge/model_10000 --train_filename training_10000 --validation_filename validation_10000 --hidden_size $STATE_DIMS --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS | tee java_logs/merge_10000.log &
