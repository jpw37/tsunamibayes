HOW TO RUN THE TSUNAMI BAYES METHOD:

----------------------- BEFORE RUNNING THE PROGRAM -----------------------

Navigate to the PreRun Folder and open the Guages.ipynb file.

This has all of the necessary functions for generating
the guages and the FG MAX data files needed to run the program.

Fill the notebook with your custom guage recordings for your specified event and run all the cells.

--------------------------  RUNNING THE PROGRAM --------------------------

Next in the Command Line navigate to the Model_Scripts/ folder.

Run python Main.py

If you give it no command line arguments it will run with the default settings as given in the Scenario Class:

title="Default_Title",
use_custom=False,
init='manual',
rw_covariance=1.0,
method="random_walk",
iterations=10

You can modify this as you please or entering the arguments in the command line as follows:

title = sys.argv[1]
use_custom = sys.argv[2]
init = sys.argv[3]
rw_covariance = sys.argv[4]
method = sys.argv[5]
iterations = int(sys.argv[6]


ex. : python Main.py 1852 True random 1.0 random_walk 1000000

This will run the program a million times initialized with random parameters and
with the methods from the Custom Class.

--------------------------- Custom Class ---------------------------

The Classes/Custom.py class is made so that any change you want to or need to make to the program
concerning the mcmc methods or the parameter space can be dealt with inside the Custom Class with no need to
interfere with the code as a whole.

***Please do not modify the custom class unless you really know what you are doing***

----------------------- Dealing With Output -----------------------

The output for this program is complied into the Model_Scripts/ModelOutput Folder as 4 files:

title_mcmc.csv - The master debug file
title_smaples - The sampled parameter space parameters for each winning sample.
                (This will be the same as okada is custom is not used)
title_okada.csv - The full okada parameters for each winning sample
title_observations - Misc


The Classes/Samples.py class is built to save and interface with the output data.
Most methods for building graphs, charts or pictures will be found in this class.

Feel free to take a look at all of the functions or add to the class if you feel not everything you need to graph
is specified in the class functions already.