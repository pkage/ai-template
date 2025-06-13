#! /bin/bash
# aiai valluvar download

#SBATCH --partition=AIAI_GPU
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --output=/home/s1734411/experiment/stdout.txt
#SBATCH --error=/home/s1734411/experiment/stderr.txt
#SBATCH --time=12:00:00
#SBATCH --job-name=experiment
#SBATCH --mail-user=p.kage@ed.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/s1734411/experiment/

export PROJ_DIR=/home/s1734411/experiment/
export LSCRATCH=/disk/scratch/s1734411/experiment


