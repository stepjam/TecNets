#!/bin/bash

if [ "$1" = "sim_reach" ]; then
    python3 main_il.py \
    --dataset=sim_reach --iterations=400000 --batch_size=64 \
    --lr=0.0005 --support=2 --query=2 --embedding=20 \
    --activation=elu --filters=40,40,40 --kernels=3,3,3 --strides=2,2,2 \
    --fc_layers=200,200,200,200 --lambda_embedding=1.0 \
    --lambda_support=0.1 --lambda_query=0.1 --margin=0.1 --norm=layer \
    --logdir='mylog/' --eval=True
elif [ "$1" = "sim_push" ]; then
    python3 main_il.py \
    --dataset=sim_push --iterations=400000 --batch_size=100 \
    --lr=0.0005 --support=5 --query=5 --embedding=20 \
    --activation=elu --filters=16,16,16,16 --kernels=5,5,5,5 \
    --strides=2,2,2,2 --fc_layers=200,200,200 --lambda_embedding=1.0 \
    --lambda_support=0.1 --lambda_query=0.1 --margin=0.1 --norm=layer \
    --logdir='mylog/' --eval=True
else
    echo 'Invalid.'
fi