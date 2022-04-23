import pandas as pd
import factor_analysis as fa

d   = pd.read_csv("data08-01.csv",index_col=0) #データ読み込み
efa = fa.ExploratoryFactorAnalysis() #インスタンス作成
nf  = efa.explore(d) # 因子数推定
fs  = efa.analyze(d,nf) # 因子分析
print(efa) # 分析結果表示

