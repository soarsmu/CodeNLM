BATCH_SIZE=128 # This is also the default.

CUDA_VISIBLE_DEVICES=4 python code_nlm.py --test True --data_path data/c-corpus/bigcode --train_dir data/c-corpus/bigcode/model_2000 --test_filename test_2000 --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True | tee c_logs/ent_bpe_2000.log &

CUDA_VISIBLE_DEVICES=5 python code_nlm.py --test True --data_path data/c-corpus/bigcode --train_dir data/c-corpus/bigcode/model_5000 --test_filename test_5000 --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True | tee c_logs/ent_bpe_5000.log &

CUDA_VISIBLE_DEVICES=6 python code_nlm.py --test True --data_path data/c-corpus/bigcode --train_dir data/c-corpus/bigcode/model_10000 --test_filename test_10000 --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True | tee c_logs/ent_bpe_10000.log &

CUDA_VISIBLE_DEVICES=6 python code_nlm.py --test True --data_path data/c-corpus/split --train_dir data/c-corpus/split/model_2000 --test_filename test_2000 --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True | tee c_logs/ent_split_2000.log &

CUDA_VISIBLE_DEVICES=5 python code_nlm.py --test True --data_path data/c-corpus/split --train_dir data/c-corpus/split/model_5000 --test_filename test_5000 --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True | tee c_logs/ent_split_5000.log &

CUDA_VISIBLE_DEVICES=4 python code_nlm.py --test True --data_path data/c-corpus/split --train_dir data/c-corpus/split/model_10000 --test_filename test_10000 --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True | tee c_logs/ent_split_10000.log &

CUDA_VISIBLE_DEVICES=5 python code_nlm.py --test True --data_path data/c-corpus/merge --train_dir data/c-corpus/merge/model_2000 --test_filename test_2000 --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True | tee c_logs/ent_merge_2000.log &

CUDA_VISIBLE_DEVICES=4 python code_nlm.py --test True --data_path data/c-corpus/merge --train_dir data/c-corpus/merge/model_5000 --test_filename test_5000 --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True | tee c_logs/ent_merge_5000.log &

CUDA_VISIBLE_DEVICES=6 python code_nlm.py --test True --data_path data/c-corpus/merge --train_dir data/c-corpus/merge/model_10000 --test_filename test_10000 --gru True --batch_size $BATCH_SIZE --word_level_perplexity True --cross_entropy True | tee c_logs/ent_merge_10000.log &

