#!/bin/bash --login


## Submission flags (Customize for VT/BYU) ##
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --time=00:05:00
#SBATCH --mem=32G
#SBATCH --mail-user=jake@math.byu.edu
#SBATCH --mail-type=END
#SBATCH --job-name=variable_start_idx
export OMP_NUM_THREADS=1
#### SETUP ####
workdir=$( pwd )
# check if nsamp is set
if [[ -z $nsamp ]]; then
    echo "Must specify a number of samples (\$nsamp)"
    exit 1
fi

# check if $cfg is set or if we are resuming a chain
if [[ -z $cfg && -z $resdir ]]; then
    echo "Must specify config file (\$cfg) or directory to resume from (\$resdir)"
    exit 1
fi
# Load software modules for installing Geoclaw (Customize for VT/BYU) ##JPW: uncomment these
# export PYTHONPATH="/fslgroup/fslg_tsunami/justin/apps/anaconda-5.2.0/lib/python3.6/site-packages"
# export PATH="/fslgroup/fslg_tsunami/justin/apps/anaconda-5.2.0/bin:$PATH"
# export PYTHONPATH="$PYTHONPATH:/fslgroup/fslg_tsunami/hmc/tsunamibayes"

conda activate new_test

# define the TMPFS environment variable if it isn't already defined (e.g., at BYU)
export TMPFS="/tmp/$SLURM_JOB_ID"; mkdir -p $TMPFS

pyargs="-v"
# settings for resumed chains
if [[ ! -z $resdir ]]; then
    resdir=$(readlink -m $resdir )
    scendir=$resdir
    rundir=$resdir
    pyargs="$pyargs -r"
fi

# settings for config files
[[ ! -z $cfg ]] && pyargs="$pyargs --cfg $( basename $cfg )"

# set jobid (e.g., "6135127_m8") and $scendir/$rundir if not already set
jobid="$( echo $SLURM_JOB_ID | sed 's/\..*$//' )_$( hostname | grep -o ^.. )"
[[ -z $scendir ]] && scendir=/fslgroup/fslg_tsunami/hmc/tsunamibayes/scenarios/banda_1852_grid
[[ -z $rundir ]] && rundir="/fslgroup/fslg_tsunami/runs/hmc_banda_1852_grid/$jobid"

logfile="$rundir/run.log"

# Get scenario-specific files for new chains
if [[ -z $resdir ]]; then
    mkdir -p $rundir
    touch $logfile
    echo "Copying scenario files to $rundir..." | tee -a $logfile
    cp -r $scendir/* $rundir
fi

# Get configfile
[[ ! -z $cfg ]] && cp $cfg $rundir

echo "resdir: $resdir"
echo "cfg: $cfg"
echo "scendir: $scendir"
echo "rundir: $rundir"
echo "logfile: $logfile"
echo "pyargs: $pyargs"
conda list
#### INSTALL CLAWPACK IN TMPFS ####
clawver=v5.9.0
# Location of clawpack source (Customize for VT/BYU)
src=/fslgroup/fslg_tsunami/jake/clawpack-$clawver.tar.gz
export CLAW="$TMPFS/clawpack-$clawver"
echo "Installing Clawpack in $CLAW..." | tee -a $logfile
mkdir -p $CLAW
cd $CLAW
tar xzf $src -C . --strip=1
export PYTHONUSERBASE=$CLAW
pip install --no-dependencies -e .  


### #RUN ####
# set some environment variables for Geoclaw
export FFLAGS='-O2 -fPIC -fopenmp -fdefault-integer-8' ; echo "FFLAGS=$FFLAGS"      | tee -a $logfile
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE; echo "OMP_NUM_THREADS=$OMP_NUM_THREADS" | tee -a $logfile
export OMP_STACKSIZE=8192M           ; echo "OMP_STACKSIZE=$OMP_STACKSIZE"          | tee -a $logfile
ulimit -s unlimited                ; echo "stack size unlimited"                    | tee -a $logfile
export OMP_NUM_THREADS=1
echo "LOG: $( date ): MCMC run start"    | tee -a $logfile
cd $rundir
# python main.py $pyargs $nsamp	| tee -a $logfile
echo "Working in directory $(pwd)"

# Restore PATH and PYTHONPATH to original to run script
# Run script
echo "FINAL WORKING DIRECTORY"
pwd
echo "CONTENTS"
ls

export PYTHONPATH="$PYTHONPATH:/flsgroup/fslg_tsunami/hmc/tsunamibayes"
echo "----------------------------------------"
echo "USING PYTHON PATH:::::"
echo $PYTHONPAH
python gradient_comps.py $pyargs $nsamp | tee -a $logfile

echo "LOG: $( date ): MCMC run complete" | tee -a $logfile

cd $workdir