#!/bin/bash

#SBATCH --job-name=JOB_NAME
#SBATCH --nodes=8
#SBATCH --mem-per-cpu=0
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --time=10-0


## change to your cluster requirements
num_additional_nodes=7
num_gpus=1  # per node


## change to your cluster requirements
source /etc/profile
module load anaconda/2020b
source activate tf
export CUDA_VISIBLE_DEVICES=0
unset PYTHONPATH

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

ulimit -u 10000
ulimit -n 10000


nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=($nodes)
echo "Current nodes:"
echo $nodes
node1=${nodes_array[0]}
echo "Head Node:"
echo $node1

suffix=':6379'
ip_head=`hostname`$suffix


export ip_head # Exporting for latter access by your command

echo "ip_head"
echo $ip_head

# ===== Start the head node =====

srun --nodes=1 -w $node1 ray start --block --head --temp-dir="/tmp/ray2/" --redis-port=6379 --num-gpus=$num_gpus & # Starting the head
sleep 10
echo "started head"
# ===== Start worker node =====
for ((i = 1; i <= $num_additional_nodes; i++)); do
  node2=${nodes_array[$i]}
  srun --nodes=1 -w $node2 ray start --block --address=$ip_head --num-gpus=$num_gpus &# Starting the workers
  sleep 10
  echo "started worker"
done

python -u driver_steps.py --cluster
