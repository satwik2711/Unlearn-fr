import numpy as np
def clustered_bootstrap(values,authors,seed=42,n=5000):
 rng=np.random.default_rng(seed);u=np.unique(authors); means=[]
 for _ in range(n):
  pick=rng.choice(u,len(u),replace=True);means.append(np.mean([np.mean(values[authors==x]) for x in pick]))
 return {'mean':float(np.mean(values)),'median':float(np.median(values)),'ci95':[float(np.quantile(means,.025)),float(np.quantile(means,.975))],'n_authors':len(u)}
