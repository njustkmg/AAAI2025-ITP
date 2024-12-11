DataSet=$1
Model=$2
METHOD=$3
GPUID=$4

cd ..
CUDA_VISIBLE_DEVICES=$4 python test_ood.py \
    --model ${Model} \
    --batch 200 \
    --dataset ${DataSet} \
    --method ${METHOD}

# sh ./test.sh CIFAR10 DN101 ITP 4
