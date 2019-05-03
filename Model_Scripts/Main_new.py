# """
# Run the scenario
# """
import argparse
import os
from datetime import datetime


## SETUP: PARSE ARGUMENTS AND SET UP RUN DIRECTORY ##

#set up command line arguments
parser = argparse.ArgumentParser(description='Run a tsunamibayes scenario.')
parser.add_argument('--scen', dest='scenario', default='1852',
                   help='scenario to run (default: 1852)')
parser.add_argument('--mcmc', dest='mcmc', default='random_walk',
                   help='mcmc method to use (default: random_walk)')
#parser.add_argument('--nburn', dest='nburn', default=0,
#                   help='number of burn in samples (default: 0)')
parser.add_argument('--nsamp', dest='nsamp', default=1,
                   help='number of samples (default: 1)')
parser.add_argument('--init', dest='init', default='rand',
                   help='initial sample: rand, manual, restart (default: rand)')
parser.add_argument('--rundir', dest='rundir', default='default',
                   help='directory to run from (default: create unique directory with scenario name)')
parser.add_argument('--runbase', dest='runbase', default='../../runs',
                   help='base directory for rundir (default: ../../runs)')

#parse command line arguments
args = parser.parse_args()

#check if the scenario exists
scenDir='Scenarios/'+args.scenario
if not os.path.exists(scenDir):
    raise Exception('Scenario directory does not exist: '+scenDir)

#set the run directory
if args.rundir == 'default':
    args.rundir = args.runbase+'/'+args.scenario+'_'+datetime.now().strftime("%Y%m%d_%H%M%S")

    #if rundir already exists, add numbers at the end
    dirName = args.rundir
    count = 1
    while os.path.exists(dirName):
        print("Directory " , dirName ,  " already exists")
        dirName = args.rundir+'_'+str(count)
        count +=1
    args.rundir = dirName
    
#create, set up, and move to the run directory
print("Running from directory ", args.rundir)
os.makedirs(args.rundir) #make directory
os.system("cp -r Classes "+args.rundir+"/")       #copy classes
os.system("cp -r "+scenDir+"/* "+args.rundir+"/") #copy scenario
os.chdir(args.rundir)


## RUN THE SCENARIO ##

import sys
sys.path.append('./Classes')

from Scenario import Scenario

#run the scenario (this needs to be finished)
##old version: scenario = Scenario(inputs['title'], inputs['custom'], inputs['init'], inputs['rw_covariance'], inputs['method'], inputs['iterations'])
#scenario = Scenario(args.scenario, args.init, inputs['rw_covariance'], args.mcmc, args.nsamp)
#scenario.run()
