#!/usr/bin/env/ python

import numpy as np
import pandas as pd
#TODO : This is another area of focus, how the program will resample over the various faults. 
def resample(output_dirs, overwrite=False,  verbose=False):
    """Reads the sample data, takes another random sample, and then writes the resampling to a .csv file.
    
    Parameters
    ----------
    output_dirs : (list) of strings
        The list of the output directories specified in the command line that store the sample's output data. 
    overwrite : bool
        The boolean flag that indicates to the function whether to overwrite the current data file or not.
        Default is false. 
    verbose : bool
        The verbose flag that prints the resample results when True. Default is false. 
    """
    samples = [pd.read_csv(dir_+"output/samples.csv", index_col=0).reset_index(drop=True) 
               for dir_ in output_dirs]
    final_idx = [df.index[-1] for df in samples]
    model_params = [pd.read_csv(dir_+"output/model_params.csv", index_col=0).reset_index(drop=True)
                    for dir_ in output_dirs]
    model_output = [pd.read_csv(dir_+"output/model_output.csv", index_col=0).reset_index(drop=True) 
                    for dir_ in output_dirs]
    bayes_data = [pd.read_csv(dir_+"output/bayes_data.csv", index_col=0).reset_index(drop=True) 
                  for dir_ in output_dirs]
    debug = [pd.read_csv(dir_+"output/debug.csv", index_col=0).reset_index(drop=True) 
             for dir_ in output_dirs]
    final_debug_idx = [df.index[-1] for df in debug]

    p = np.exp(np.array([df['posterior_logpdf'].iloc[-1] for df in bayes_data]))
    p /= p.sum()
    resample_idx = np.random.choice(np.arange(len(p)), size=len(p), p=p)
    if verbose:
        print("Probabilies:")
        for i,dir_ in enumerate(output_dirs):
            print(dir_+": "+str(p[i]))
        print("\nResample results:\n-------------------")
    
    zipped = zip(samples,model_params,model_output,
                 bayes_data,debug,resample_idx,output_dirs)
    
    for df_s,df_mp,df_mo,df_bd,df_d,idx,dir_ in zipped:
        if verbose:
            print("Chain at {}:\n".format(dir_))
            print("Resampled from {}".format(output_dirs[idx]))
            print(samples[idx].loc[final_idx[idx]],"\n")
       
        if overwrite:
            df_s.loc[len(df_s)] = samples[idx].loc[final_idx[idx]]
            df_mp.loc[len(df_mp)] = model_params[idx].loc[final_idx[idx]]
            df_mo.loc[len(df_mo)] = model_output[idx].loc[final_idx[idx]]
            df_bd.loc[len(df_bd)] = bayes_data[idx].loc[final_idx[idx]]

            metro_hastings_data = pd.Series({'alpha':1,'accepted':1,
                                             'acceptance_rate':np.nan})
            df_d.loc[len(df_d)] = gen_debug_row(df_s.iloc[-2],df_s.iloc[-1],
                                                df_mp.iloc[-2],df_mp.iloc[-1],
                                                df_bd.iloc[-2],df_bd.iloc[-1],
                                                metro_hastings_data)
            df_s.to_csv(dir_+"output/samples.csv")
            df_mp.to_csv(dir_+"output/model_params.csv")
            df_mo.to_csv(dir_+"output/model_output.csv")
            df_bd.to_csv(dir_+"output/bayes_data.csv")
            df_d.to_csv(dir_+"output/debug.csv")

def gen_debug_row(sample,proposal,sample_model_params,proposal_model_params,
                      sample_bayes,proposal_bayes,metro_hastings_data):
        """Combines data from a scenario iteration to be used for debugging puposes, and 
        stores the combined data in a Pandas Series object.

        Parameters
        ----------
        sample : pandas Series of floats
            The series that contains the float values for the declared sample columns.
        proposal : pandas Series of floats
            The series that contains the float values for the declared proposal columns.
        sample_model_params : dict
            The dictionary containing the sample's specified parameters and their associated float values. 
            This differs from subclass to subclass, but generally contains parameters such as 
            magnitude, length, width, etc. etc.
        proposal_model_params :  dict
            The dictionary containing the proposal's specified parameters. This differs from subclass
            to subclass, but generally contains parameters such as magnitude, length, width, etc. etc.
        sample_bayes : pandas Series
            The pandas series that contains labels and float values for 
            the sample's prior logpdf, loglikelihood, and posterior logpdf.
        proposal_bayes : pandas Series
            The pandas series that contains labels and float values for 
            the prosal's prior logpdf, loglikelihood, and posterior logpdf.
        metro_hastings_data : pandas Series
            The series that contains a dictioanry with the acceptace probablity, acceptace state, and acceptance rate.

        Returns
        -------
        debug_row : pandas Series
            The combined/concatenated series of all the dictionaries and pd.series passed in to the function.
        """
        proposal = pd.Series(proposal).rename(lambda x:'p_'+x)
        sample_model_params = pd.Series(sample_model_params).rename(
                              lambda x:'m_'+x if x in sample.index else x)
        proposal_model_params = pd.Series(proposal_model_params).rename(
                                lambda x:'p_m_'+x if x in sample.index else 'p_'+x)
        proposal_bayes = pd.Series(proposal_bayes).rename(lambda x:'p_'+x)
        return pd.concat((sample,
                          proposal,
                          sample_model_params,
                          proposal_model_params,
                          sample_bayes,
                          proposal_bayes,
                          metro_hastings_data))

if __name__ == "__main__":
    """Executes the resampling function with the list of command line prompts."""
    from sys import argv
    if argv[1] == '-o':
        resample(argv[2:], overwrite=True, verbose=True)
    else:
        resample(argv[1:],verbose=True)
