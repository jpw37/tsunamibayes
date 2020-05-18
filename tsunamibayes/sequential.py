import numpy as np
import pandas as pd

def resample(output_dirs, verbose=False):
    sample_data = [pd.read_csv(dir+"samples.csv", index_col=0) for dir in output_dirs]
    model_param_data = [pd.read_csv(dir+"model_params.csv", index_col=0) for dir in output_dirs]
    model_output_data = [pd.read_csv(dir+"model_output.csv", index_col=0) for dir in output_dirs]
    bayes_data = [pd.read_csv(dir+"bayes_data.csv", index_col=0) for dir in output_dirs]

    p = np.exp(np.array([df['posterior_logpdf'].iloc[-1] for df in bayes_data]))
    p /= p.sum()
    resample_idx = np.random.choice(np.arange(len(p)), size=len(p), p=p)
    if verbose: print("Resample results:\n-------------------")

    zipped = zip(sample_data,model_param_data,model_output_data,
                 bayes_data,resample_idx,output_dirs)
    for df_s,df_mp,df_mo,df_bd,idx,dir in zipped:
        df_s.loc[len(df)] = sample_data[idx].iloc[-1]
        df_mp.loc[len(df)] = model_param_data[idx].iloc[-1]
        df_mo.loc[len(df)] = model_output_data[idx].iloc[-1]
        df_bd.loc[len(df)] = bayes_data[idx].iloc[-1]
        if verbose: print("Chain at {}:\n".format(dir),final_samples[idx],"\n")
        df_s.to_csv(dir+"samples.csv")
        df_mp.to_csv(dir+"model_params.csv")
        df_mo.to_csv(dir+"model_output.csv")
        df_bd.to_csv(dir+"bayes_data.csv")

if __name__ == "__main__":
    from sys import argv
    resample(argv[1:])
