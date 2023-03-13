# !/bin/bash

for ((batch_size = 32; batch_size <= 256; batch_size = batch_size * 2))
do
    for ((episode = 60000; episode <= 150000; episode = episode + 10000))
    do
        echo "----------  batch_size: $batch_size, episode: $episode  ----------"
        python train.py --batch_size "$batch_size" --episode "$episode"
        python eval.py --batch_size "$batch_size" --episode "$episode"
    done
done