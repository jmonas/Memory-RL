#!/bin/bash
#SBATCH --job-name=RL_Transformer     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=12G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=jmonas@princeton.edu
#SBATCH --partition=mig

module purge
module load anaconda3/2024.2
module load cudatoolkit/11.7
module load cudnn/cuda-11.x/8.2.0
conda activate /scratch/gpfs/jmonas/mambaforge/envs/mem_rl_2

python main.py \
    --config_env configs/envs/tmaze_passive.py \
    --config_env.env_name 25 \
    --config_rl configs/rl/dqn_default.py \
    --train_episodes 6700 \
    --config_seq configs/seq_models/gpt_default.py \
    --config_seq.sampled_seq_len -1 \
    --config_seq.model.seq_model_config.n_layer 1 \
    --config_seq.model.seq_model_config.n_head 1 \
