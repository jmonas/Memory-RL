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

module purge
module load cudatoolkit/11.1
module load cudnn/cuda-11.x/8.2.0
module load anaconda3/2023.9
conda activate /home/jmonas/mambaforge/envs/LLM_debate

python main.py \
    --config_env configs/envs/tmaze_active.py \
    --config_env.env_name 25 \
    --config_rl configs/rl/dqn_default.py \
    --train_episodes 40000 \
    --config_seq configs/seq_models/gpt_default.py \
    --config_seq.sampled_seq_len -1 \
    --config_seq.model.seq_model_config.n_layer 1 \
    --config_seq.model.seq_model_config.n_head 1 \

conda create --name trans_mem python=3.8
conda activate trans_mem
pip install -r requirements.txt
