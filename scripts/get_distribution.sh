DataSet=$1
Model=$2
GPUID=$3

cd ..
CUDA_VISIBLE_DEVICES=$3 python get_distribution.py \
    --dataset ${DataSet} \
    --model ${Model} \

# sh ./get_distribution.sh CIFAR10 DN101 4
