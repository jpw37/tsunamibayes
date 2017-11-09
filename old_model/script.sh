# This is a bash script for running Geoclaw with MCMC

# MODIFY HERE (number of iterations)
ITERS=1
COUNTER=0

value=0

# MUST MATCH maketopo.py line 49
rm dtopo.tt3
rm dtopo.data

python init_params.py

make clean
make clobber

make .output

python init_output.py

while [  $COUNTER -lt $ITERS ]; do

    # output current iteration
    echo Iteration $COUNTER Monte Carlo Method
    sleep 2s

    # remove old topo file
    # MODIFY HERE (must match maketopo.py line 49)
    rm dtopo.tt3
    rm dtopo.data

    # set parameters and make new topo
    python set_params.py

    # clean Geoclaw
    make clean
    make clobber

    # run output
    make .output

    # evaluate output
    python read_output.py

    # increment counter
    let COUNTER=COUNTER+1

done
