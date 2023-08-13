import pandas as pd
import numpy as np
#import polychoric as pc
#from polychoric import polychoric_corr
import polychoric

d   = pd.read_csv("survey_3Q.csv",index_col=0) #データ読み込み
cor = polychoric.corr(d,method="SLSQP",verbose=False)
#cor = polychoric.corr(d,method="deriv_check")

print(cor) # 分析結果表示

