import pandas as pd
import factor_analysis as fa
import sem_analysis as sa
#import semopy

d   = pd.read_csv("data08-01.csv",index_col=0) #データ読み込み
efa = fa.ExploratoryFactorAnalysis() #インスタンス作成
nf  = efa.explore(d) # 因子数推定
fs  = efa.analyze(d,nf) # 因子分析
print(efa) # 分析結果表示

print("CFA:")
cfa1 = efa.cfa_model(formating="lavaan",model_type="order1")
cfa2 = efa.cfa_model(formating="lavaan",model_type="order2")
cfab = efa.cfa_model(formating="lavaan",model_type="bifactor")
cfaf = efa.cfa_model(formating="lavaan",model_type="multilayer")
#print(cfa)

model1 = sa.StructEquationModel(cfa1)
model2 = sa.StructEquationModel(cfa2)
modelb = sa.StructEquationModel(cfab)
modelf = sa.StructEquationModel(cfaf)

model1.fit(d)
model2.fit(d)
modelb.fit(d)
modelf.fit(d)

model1.plot("sample_1.png")
model2.plot("sample_2.png")
modelb.plot("sample_b.png")
modelf.plot("sample_f.png")

print(model1)
print(model2)
print(modelb)
print(modelf)

