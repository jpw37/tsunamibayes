#!/usr/bin/env/ python

import numpy as np
import pandas as pd

def resample(output_dirs, overwrite=False,  verbose=False):
    samples = [pd.read_csv(dir+"output/samples.csv", index_col=0) for dir in output_dirs]
    final_samples = [df.iloc[-1] for df in samples]
    model_params = [pd.read_csv(dir+"output/model_params.csv", index_col=0) for dir in output_dirs]
    model_output = [pd.read_csv(dir+"output/model_output.csv", index_col=0) for dir in output_dirs]
    bayes_data = [pd.read_csv(dir+"output/bayes_data.csv", index_col=0) for dir in output_dirs]
    debug = [pd.read_csv(dir+"output/debug.csv", index_col=0) for dir in output_dirs]

    p = np.exp(np.array([df['posterior_logpdf'].iloc[-1] for df in bayes_data]))
    p /= p.sum()
    resample_idx = np.random.choice(np.arange(len(p)), size=len(p), p=p)
    if verbose:
        print("Probabilies:")
        for i,dir in enumerate(output_dirs):
            print(dir+": "+str(p[i]))
        print("\nResample results:\n-------------------")

    zipped = zip(samples,model_params,model_output,
                 bayes_data,debug,resample_idx,output_dirs)
    p_cols = ['p_'+col for col in samples[0].columns]
    for df_s,df_mp,df_mo,df_bd,df_d,idx,dir in zipped:
        if verbose:
            print("Chain at {}:\n".format(dir))
            print("Resampled from {}".format(output_dirs[idx]))
            print(final_samples[idx],"\n")
            df_d.loc[len(df_d),df_s.columns] = df_s.iloc[-1]
            df_d.loc[len(df_d),p_cols] = final_samples[idx].rename(lambda x:'p_'+x)
            df_d.loc[len(df_d),'accepted'] = 1
            df_s.loc[len(df_s)] = final_samples[idx]
            df_mp.loc[len(df_mp)] = model_params[idx].iloc[-1]
            df_mo.loc[len(df_mo)] = model_output[idx].iloc[-1]
            df_bd.loc[len(df_bd)] = bayes_data[idx].iloc[-1]
        if overwrite:
            df_s.to_csv(dir+"output/samples.csv")
            df_mp.to_csv(dir+"output/model_params.csv")
            df_mo.to_csv(dir+"output/model_output.csv")
            df_bd.to_csv(dir+"output/bayes_data.csv")
            df_d.to_csv(dir+"output/debug.csv")

if __name__ == "__main__":
    from sys import argv
    if argv[1] == '-o':
        resample(argv[2:], overwrite=True, verbose=True)
    else:
        resample(argv[1:],verbose=True)
