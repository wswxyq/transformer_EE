# make a fake data set for testing

# %%
import json
with open("transformer_ee/config/input.json", "r") as f:
    input_d = json.load(f)

# %%
import numpy as np
from random import randrange

factor = np.random.rand(14)

vname=input_d["vector"]
sname=input_d["scalar"]
tname=input_d["target"]

d={
    x:[] for x in vname+tname+sname
}

for i in range(100000):
    if i%5000==0:
        print(i)
    v = np.random.rand(randrange(1,20))
    for j, x in enumerate(vname):
        tmp=v*factor[j]
        d[x].append(np.array2string(tmp, separator=',', precision=3, prefix='', suffix='')[1:-1])
    for j, x in enumerate(sname):
        tmp=0
        d[x].append(tmp)
    
    for j, x in enumerate(tname):
        tmp=v*factor[j+12]
        d[x].append(np.sum(tmp))
    

    
# %%
# %%
import pandas as pd

df = pd.DataFrame.from_dict(d)
# %%
df.to_csv("transformer_ee/data/input.csv", index=False)
