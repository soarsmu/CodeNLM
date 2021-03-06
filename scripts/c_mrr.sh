BATCH_SIZE=10 # This is also the default.

CUDA_VISIBLE_DEVICES=3 python code_nlm.py --completion True --data_path data/c-corpus/bigcode --train_dir data/c-corpus/bigcode/model_10000 --test_filename test_10000 --identifier_map data/c-corpus/bigcode/id_test_10000 --gru True --batch_size $BATCH_SIZE | tee c_logs/mrr_bpe_10000.log &

CUDA_VISIBLE_DEVICES=4 python code_nlm.py --completion True --data_path data/c-corpus/split --train_dir data/c-corpus/split/model_10000 --test_filename test_10000 --identifier_map data/c-corpus/split/id_test_10000 --gru True --batch_size $BATCH_SIZE | tee c_logs/mrr_split_10000.log &

CUDA_VISIBLE_DEVICES=1 python code_nlm.py --completion True --data_path data/c_corpus --train_dir data/c_corpus/model_10000 --test_filename test_split_10000 --identifier_map data/c_corpus/id_test_split_10000 --gru True --batch_size 10 | tee c_logs/mrr_merge_10000.log &

