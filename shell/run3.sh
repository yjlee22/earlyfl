export CUDA_VISIBLE_DEVICES=0

# Experiment parameters
methods=("FedAvg" "FedDyn" "FedSAM" "FedGamma" "FedSpeed" "FedSMOO")
seeds=(0 1 2 3 4)
patience_values=(1 5 10)
num_data=(10 20 50 100)

for seed in "${seeds[@]}"; do
    for p in "${patience_values[@]}"; do
        for m in "${methods[@]}"; do
            for n in "${num_data[@]}"; do
                python train.py --method ${m} --non-iid --syn --generator roentgen --num_per_class ${n} --early --patience ${p} --seed ${seed}
            done
        done
    done
done