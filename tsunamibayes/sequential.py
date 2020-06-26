#!/usr/bin/env/ python

import numpy as np
import pandas as pd

def resample(output_dirs, verbose=False):
    """Reads the sample data, takes another random sample, and then writes the resampling to a .csv file.
    
    Parameters
    ----------
    output_dirs : (list) of strings
        The list of the output directories specified in the command line that store the sample's output data. 
    verbose : bool
        The verbose flag that prints the resample results when True. Default is false. 
    """
    sample_data = [pd.read_csv(dir+"samples.csv", index_col=0) for dir in output_dirs]
    final_samples = [df.iloc[-1] for df in sample_data]
    bayes_data = [pd.read_csv(dir+"bayes_data.csv", index_col=0) for dir in output_dirs]

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
        """Create a Pandas Series object with the given data that is desired to
        be kept in the debug output.

        Parameters
        ----------
        # TODO: EITHER USE **kwargs OR A BUNCH OF SPECIFIC PARAMETERS?

        Returns
        -------
        Pandas.Series
            Row for the debug Dataframe
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
