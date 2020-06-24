import numpy as np
import pandas as pd

def resample(output_dirs, verbose=False):
    sample_data = [pd.read_csv(dir+"samples.csv", index_col=0) for dir in output_dirs]
    final_samples = [df.iloc[-1] for df in sample_data]
    bayes_data = [pd.read_csv(dir+"bayes_data.csv", index_col=0) for dir in output_dirs]

    p = np.exp(np.array([df['posterior_logpdf'].iloc[-1] for df in bayes_data]))
    p /= p.sum()
    resample_idx = np.random.choice(np.arange(len(p)), size=len(p), p=p)
    if verbose: print("Resample results:\n-------------------")
    for df,idx,dir in zip(sample_data,resample_idx,output_dirs):
        df.loc[len(df)] = final_samples[idx]
        if verbose: print("Chain at {}:\n".format(dir),final_samples[idx],"\n")
        df.to_csv(path+"samples.csv")

if __name__ == "__main__":
    from sys import argv
    resample(argv[1:])
