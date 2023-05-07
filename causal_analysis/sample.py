import numpy as np
import pandas as pd

import causal_analysis as ca

data   = pd.read_csv("japan_data_scaled.csv",index_col=0,header=1) #データ読み込み
print(data)

#stddat = ca.standardize(data)
#print(stddat)
#
#nmldat = ca.normalize(data)
#print(nmldat)

scldat = ca.scale_by(data)
print(scldat)


analyzer = ca.CausalAnalysis(method="sem",alpha=0.05,enable_cov=False) # 有意水準をザルにしないと因果がなくなっちゃう

#stdmod = analyzer.discover(stddat)
#print(stdmod)
#nmlmod = analyzer.discover(nmldat)
#print(nmlmod)
#sclmod = analyzer.discover(scldat)
#print(sclmod)

#analyzer.analyze(data)
analyzer.analyze(scldat)

analyzer.save()
print(analyzer)
