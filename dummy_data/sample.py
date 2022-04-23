import dummy_data as dd
import pandas as pd

data = dd.make_dummy_likert(100,1000)
stddat = data - data.mean() / data.std(ddof=0)
    
print(data.describe())

print(data)
