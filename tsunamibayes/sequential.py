import numpy as np
import pandas as pd

def resample(output_paths):
    sample_data = [pd.read_csv(path+"samples.csv",index_col=0) for path in output_paths]
    final_samples = [df.iloc[-1] for df in sample_data]
    bayes_data = [pd.read_csv(path+"bayes_data.csv",index_col=0) for path in output_paths]

    p = np.exp(np.array([df['posterior_logpdf'].iloc[-1] for df in bayes_data]))
    p /= p.sum()
    resample_idx = np.random.choice(np.arange(len(p)),size=len(p),p=p)
    for df,idx,path in zip(sample_data,resample_idx,output_paths):
        df.loc[len(df)] = final_samples[idx]
        df.to_csv(path+"samples.csv")
