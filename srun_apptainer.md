# the following command request a V100 node in Wilson CLuster 
## (ignore the warning: bash: /nashome/.../.bashrc: Permission denied)

[Go to official website for a full list of the nodes](https://computing.fnal.gov/wilsoncluster/hardware/)

[recommanded config]
```
srun --unbuffered --pty -A nova --partition=gpu_gce \
     --time=08:00:00 \
     --nodes=1 --ntasks-per-node=1 --gres=gpu:1 \
     --nodelist wcgpu06 /bin/bash

module load apptainer

export APPTAINER_CACHEDIR=/scratch/.singularity/cache

mkdir /scratch/work

apptainer shell  --nv \
    --workdir=/scratch/work \
    --home=/work1/nova/wus/ \
    /wclustre/nova/users/wus/pytorch-1.13-py3.sif
```

# the mappped directory(--home, etc) will store the change in container i.e. .bash_history, .bash_rc 

