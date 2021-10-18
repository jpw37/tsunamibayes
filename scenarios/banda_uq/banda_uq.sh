#!/bin/bash

#### SETUP ####
workdir=$( pwd )
# check if nsamp is set
if [[ -z $nsamp ]]; then
    echo "Must specify a number of samples (\$nsamp)"
    return
fi

# check if $cfg is set or if we are resuming a chain
if [[ -z $cfg && -z $resdir ]]; then
    echo "Must specify config file (\$cfg) or directory to resume from (\$resdir)"
    return
fi

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
jobid="test_UQ_banda"
# Set scendir and rundir to the places where you want to actually save them.
[[ -z $scendir ]] && scendir="/Users/paskett/tsunamibayes/scenarios/banda_uq/"
[[ -z $rundir ]] && rundir="/Users/paskett/Desktop/thesis/uncertainty_quantification/UQ_runs/$jobid/"

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

#### RUN ####
echo "LOG: $( date ): MCMC run start" | tee -a $logfile
cd $rundir
python main.py $pyargs $nsamp	| tee -a $logfile

echo "LOG: $( date ): MCMC run complete" | tee -a $logfile
cd $workdir
