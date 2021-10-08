subword-nmt learn-bpe -s 2000 < ./data/java-corpus/merge/merge > ./data/java-corpus/merge/encoding.merge_2000 &

subword-nmt learn-bpe -s 5000 < ./data/java-corpus/merge/merge > ./data/java-corpus/merge/encoding.merge_5000 &

subword-nmt learn-bpe -s 10000 < ./data/java-corpus/merge/merge > ./data/java-corpus/merge/encoding.merge_10000

subword-nmt apply-bpe -c ./data/java-corpus/merge/encoding.merge_2000 < ./data/java-corpus/bigcode/java_training_pre > ./data/java-corpus/merge/java_training_pre_merge_2000 &

subword-nmt apply-bpe -c ./data/java-corpus/merge/encoding.merge_5000 < ./data/java-corpus/bigcode/java_training_pre > ./data/java-corpus/merge/java_training_pre_merge_5000 &

subword-nmt apply-bpe -c ./data/java-corpus/merge/encoding.merge_10000 < ./data/java-corpus/bigcode/java_training_pre > ./data/java-corpus/merge/java_training_pre_merge_10000 &

subword-nmt apply-bpe -c ./data/java-corpus/merge/encoding.merge_2000 < ./data/java-corpus/bigcode/java_test_pre > ./data/java-corpus/merge/test_merge_2000 &

subword-nmt apply-bpe -c ./data/java-corpus/merge/encoding.merge_5000 < ./data/java-corpus/bigcode/java_test_pre > ./data/java-corpus/merge/test_merge_5000 &

subword-nmt apply-bpe -c ./data/java-corpus/merge/encoding.merge_10000 < ./data/java-corpus/bigcode/java_test_pre > ./data/java-corpus/merge/test_merge_10000 &

subword-nmt apply-bpe -c ./data/java-corpus/merge/encoding.merge_2000 < ./data/java-corpus/bigcode/java_validation_pre > ./data/java-corpus/merge/validation_merge_2000 &

subword-nmt apply-bpe -c ./data/java-corpus/merge/encoding.merge_5000 < ./data/java-corpus/bigcode/java_validation_pre > ./data/java-corpus/merge/validation_merge_5000 &

subword-nmt apply-bpe -c ./data/java-corpus/merge/encoding.merge_10000 < ./data/java-corpus/bigcode/java_validation_pre > ./data/java-corpus/merge/validation_merge_10000

# Maximum training epochs
EPOCHS=50
# Initial learning rate
LR=0.1 # This is the default value. You can skip it if you don't want to change it.
# Training batch size
BATCH_SIZE=32 # This is also the default.
# RNN unroll timesteps for gradient calculation.
STEPS=200 # This also the default.
# 1 - Dropout probability
KEEP_PROB=0.5 # This is also the default.
# RNN hidden state size
STATE_DIMS=512 # This is also the default.
# Checkpoint and validation loss calculation frequency.
CHECKPOINT_EVERY=5000 # This is also the default.

# Train a small py model for 1 epoch.
CUDA_VISIBLE_DEVICES=1 python code_nlm.py --data_path data/java-corpus/merge --train_dir data/java-corpus/merge/model_2000 --train_filename java_training_pre_merge_2000 --validation_filename validation_merge_2000 --gru True --hidden_size $STATE_DIMS  --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS | tee java_logs/bpe_split_merge_2000.log &

CUDA_VISIBLE_DEVICES=2 python code_nlm.py --data_path data/java-corpus/merge --train_dir data/java-corpus/merge/model_5000 --train_filename java_training_pre_merge_5000 --validation_filename validation_merge_5000 --gru True --hidden_size $STATE_DIMS  --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS | tee java_logs/bpe_split_merge_5000.log &

CUDA_VISIBLE_DEVICES=3 python code_nlm.py --data_path data/java-corpus/merge --train_dir data/java-corpus/merge/model_10000 --train_filename java_training_pre_merge_10000 --validation_filename validation_merge_10000 --gru True --hidden_size $STATE_DIMS  --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True --steps_per_checkpoint $CHECKPOINT_EVERY --max_epoch $EPOCHS | tee java_logs/bpe_split_merge_10000.log &