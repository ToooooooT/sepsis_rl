# !/bin/bash

episode = 160000
for ((batch_size = 32; batch_size <= 256; batch_size = batch_size * 2))
do
    for lr in {1..10..2}
    do
        for reg_lambda in {0..5}
        do
            echo "----------  batch_size: $batch_size, episode: $episode, use_priority: True, reg_lambda: $reg_lambda, learning rate: $lr ----------"
            python train.py --batch_size "$batch_size" --episode "$episode" --use_pri "1" --reg_lambda "$reg_lambda" --lr "$lr"
            python eval.py --batch_size "$batch_size" --episode "$episode" --use_pri "1" --reg_lambda "$reg_lambda" --lr "$lr"
        done
    done
done