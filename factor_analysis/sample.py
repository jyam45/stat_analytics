import pandas as pd
import factor_analysis as fa
#import sem_analysis as sa
import semopy

d   = pd.read_csv("data08-01.csv",index_col=0) #データ読み込み
efa = fa.ExploratoryFactorAnalysis() #インスタンス作成
nf  = efa.explore(d) # 因子数推定
fs  = efa.analyze(d,nf) # 因子分析
print(efa) # 分析結果表示

print("CFA model:")
cfa = efa.cfa_model(form="lavaan")
print(cfa)

#cfi = StructEquationModel(cfa)
mod = semopy.Model(cfa)
res = mod.fit(d)
print(res)

ins = mod.inspect()
pd.set_option('display.max_columns',len(ins.columns))
pd.set_option('display.max_rows',len(ins.index))
print(ins)

gfi = semopy.calc_stats(mod)
pd.set_option('display.max_columns',len(gfi.index))
pd.set_option('display.max_rows',len(gfi.columns))
print(gfi.T)

print("Graphviz:")
g = semopy.semplot(mod,"dat.png",plot_covs=True,std_ests=True,show=True)
print(g)

