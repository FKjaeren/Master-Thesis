#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J MetaHeuristic
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that the job must run on this type of CPU --
#BSUB -R "select[model == XeonE5_2650v4]"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=2GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 3GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 60:00 
### -- choose email address to notify when complete -- 
#BSUB -u s174478@student.dtu.dk
### -- send notification at start -- 
###BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -oo Output.out 
#BSUB -oe Error.err

# here follow the commands you want to execute
python DataPrep.py &> result.txt
