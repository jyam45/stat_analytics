import pandas as pd
import corr_analysis as ca

d   = pd.read_csv("data08-01.csv",index_col=0) #データ読み込み
cor = ca.CorrelationAnalysis() #インスタンス作成
#cor = ca.CorrelationAnalysis(partial=True) #インスタンス作成
sigs= cor.analyze(d) # 相関分析
print(cor) # 分析結果表示

