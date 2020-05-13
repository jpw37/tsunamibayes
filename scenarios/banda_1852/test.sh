#!/bin/bash


## Submission flags (Customize for VT/BYU) ##
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=4096M #memory requirement
#SBATCH --mail-user=hringer@mathematics.byu.edu   # email address
#SBATCH --mail-type=END
#SBATCH --qos=test


#### SETUP ####
## Load software modules (Customize for VT/BYU) ##
export PYTHONPATH="/fslgroup/fslg_tsunami/justin/apps/anaconda-5.2.0/lib/python3.6/site-packages:$PYTHONPATH"
export PATH="/fslgroup/fslg_tsunami/justin/apps/anaconda-5.2.0/bin:$PATH"

#define the TMPFS environment variable if it isn't already defined (e.g., at BYU)
export TMPFS="/tmp/$SLURM_JOB_ID"; mkdir -p $TMPFS

#e.g., "520404_br"
jobid="$( echo $SLURM_JOB_ID | sed 's/\..*$//' )_$( hostname | grep -o ^.. )"

#parse arguments
[[ -z $configfile ]] && configfile="defaults.cfg"
[[ -z $init       ]] && init="manual"
[[ -z $workdir ]]  && workdir=$( pwd )
[[ -z $rundir ]]   && rundir="$TMPFS/banda_1852"
[[ -z $finaldir ]] && finaldir="/fslhome/hringer/fsl_groups/fslg_tsunami/compute/hringer/runs/$jobid"

# sbatch --export="pyargs=\"\""

logfile="$TMPFS/run.log"

#if being run with gnu parallel, create a subdirectory of finaldir for each
[[ ! -z $PARALLEL_SEQ ]] && finaldir="$finaldir/$( printf %03d $PARALLEL_SEQ )"

#### INSTALL CLAWPACK IN TMPFS ####
clawver=v5.6.0
#Location of clawpack source (Customize for VT/BYU)
src=/fslgroup/fslg_tsunami/justin/src/clawpack-$clawver.tar.gz
export CLAW="$TMPFS/clawpack-$clawver"
mkdir -p $CLAW
cd $CLAW
tar xzf $src -C . --strip=1
export PYTHONUSERBASE="$PYTHONUSERBASE:$CLAW"
python setup.py install --user
mkdir $rundir
cd $rundir
cp -r $workdir $TMPFS
ls

#### RUN ####

#initialize log file
> $logfile

#set some environment variables for Geoclaw
export FFLAGS='-O2 -fPIC -fopenmp' ; echo "FFLAGS=$FFLAGS"                          | tee -a $logfile
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE; echo "OMP_NUM_THREADS=$OMP_NUM_THREADS" | tee -a $logfile
export OMP_STACKSIZE=16M           ; echo "OMP_STACKSIZE=$OMP_STACKSIZE"            | tee -a $logfile
ulimit -s unlimited                ; echo "stack size unlimited"                    | tee -a $logfile

echo "LOG: $( date ): MCMC run start"    | tee -a $logfile

#Customize: Set run parameters
python main.py --cfg test.cfg -v 5 | tee -a $logfile
#TODO: Add command line arguments
#    --scen    1852mag      \
#    --mcmc    random_walk  \
#    --nsamp   500         \
#    --rundir  $rundir      \
#    --init manual          \

echo "LOG: $( date ): MCMC run complete" | tee -a $logfile

#### PLOT ####

# echo "LOG: $( date ): Plotting/summarizing..." | tee -a $logfile
# #TODO: Add plotting
# #TODO: Add summary (accept/reject ratio, MAP and MLE points, etc)
# echo "LOG: $( date ): Plotting complete" | tee -a $logfile


#### SAVE RESULTS ####

#copy from rundir to final directory
mkdir -p $finaldir
cp -r $rundir $finaldir
cp $logfile $finaldir

#change back to working directory
cd $workdir
