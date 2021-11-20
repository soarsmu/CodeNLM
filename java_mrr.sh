BATCH_SIZE=10 # This is also the default.

CUDA_VISIBLE_DEVICES=4 python code_nlm.py --completion True --data_path data/java-corpus/bigcode --train_dir data/java-corpus/bigcode/model_2000 --test_filename test_2000 --identifier_map data/java-corpus/bigcode/id_test_2000 --gru True --batch_size $BATCH_SIZE | tee java_logs/mrr_bpe_2000.log &

CUDA_VISIBLE_DEVICES=5 python code_nlm.py --completion True --data_path data/java-corpus/bigcode --train_dir data/java-corpus/bigcode/model_5000 --test_filename test_5000 --identifier_map data/java-corpus/bigcode/id_test_5000 --gru True --batch_size $BATCH_SIZE | tee java_logs/mrr_bpe_5000.log &

CUDA_VISIBLE_DEVICES=6 python code_nlm.py --completion True --data_path data/java-corpus/bigcode --train_dir data/java-corpus/bigcode/model_10000 --test_filename test_10000 --identifier_map data/java-corpus/bigcode/id_test_10000 --gru True --batch_size $BATCH_SIZE | tee java_logs/mrr_bpe_10000.log &

CUDA_VISIBLE_DEVICES=6 python code_nlm.py --completion True --data_path data/java-corpus/split --train_dir data/java-corpus/split/model_2000 --test_filename test_2000 --identifier_map data/java-corpus/split/id_test_2000 --gru True --batch_size $BATCH_SIZE | tee java_logs/mrr_split_2000.log &

CUDA_VISIBLE_DEVICES=4 python code_nlm.py --completion True --data_path data/java-corpus/split --train_dir data/java-corpus/split/model_5000 --test_filename test_5000 --identifier_map data/java-corpus/split/id_test_5000 --gru True --batch_size $BATCH_SIZE | tee java_logs/mrr_split_5000.log &

CUDA_VISIBLE_DEVICES=5 python code_nlm.py --completion True --data_path data/java-corpus/split --train_dir data/java-corpus/split/model_10000 --test_filename test_10000 --identifier_map data/java-corpus/split/id_test_10000 --gru True --batch_size $BATCH_SIZE | tee java_logs/mrr_split_10000.log &

CUDA_VISIBLE_DEVICES=5 python code_nlm.py --completion True --data_path data/java-corpus/merge --train_dir data/java-corpus/merge/model_2000 --test_filename test_2000 --identifier_map data/java-corpus/merge/id_test_2000 --gru True --batch_size $BATCH_SIZE | tee java_logs/mrr_merge_2000.log &

CUDA_VISIBLE_DEVICES=4 python code_nlm.py --completion True --data_path data/java-corpus/merge --train_dir data/java-corpus/merge/model_5000 --test_filename test_5000 --identifier_map data/java-corpus/merge/id_test_5000 --gru True --batch_size $BATCH_SIZE | tee java_logs/mrr_merge_5000.log &

CUDA_VISIBLE_DEVICES=6 python code_nlm.py --completion True --data_path data/java-corpus/merge --train_dir data/java-corpus/merge/model_10000 --test_filename test_10000 --identifier_map data/java-corpus/merge/id_test_10000 --gru True --batch_size $BATCH_SIZE | tee java_logs/mrr_merge_10000.log &

CUDA_VISIBLE_DEVICES=1 python code_nlm.py --completion True --data_path data/c/split --train_dir data/c/split/model --test_filename test --identifier_map data/c/split/id_test --gru True --batch_size 10 | tee c_logs/test_split.log
