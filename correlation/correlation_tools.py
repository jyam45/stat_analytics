import itertools as itr
import pandas as pd
import scipy as sp
import numpy as np
 
# 有意な相関のリスト
#  df : pandas dataFrame format
#  alpha : max of p-value
def corlist(df,alpha=0.05):
    clist=[]
    for i,j in itr.combinations(df,2):
        x = df.loc[:,[i]].values
        y = df.loc[:,[j]].values
        r,p = sp.stats.pearsonr(np.ravel(x),np.ravel(y))
        if p < alpha :
            cpair = { 'first':i, 'second':j, 'r':r, 'p':p }
            clist.append(cpair)
    return clist

#p値つき編相関係数行列の計算
#   df : pandas.DataFrame
def pcorr(df):
    cor = df.corr()
    n   = df.shape[0] # number of data
    l   = df.shape[1] # number of labels
    # 偏相関係数の計算
    cor_inv = sp.linalg.pinv(cor) # 従属変数が存在する場合に備えて、擬逆行列pinvを使用
    denom = 1 / np.sqrt(np.diag(cor_inv))
    numer = np.repeat(denom,l).reshape(l,l)
    pcor  = (-cor_inv) * denom * numer
    np.fill_diagonal(pcor,0) #ゼロ割防止のため対角成分を０にする
    # 統計検定量(t)の計算
    t = np.abs(pcor) * np.sqrt(n-3) / np.sqrt(1-pcor*pcor)#３変数間の相関なのでn-3
    np.fill_diagonal(pcor,1) #対角成分を自己相関値=1にする
    # p値の計算
    p = 2*sp.stats.t.cdf(-t,n-3) #両側、Array型
    # pandas.DataFrameへ変換
    pcor_df = pd.DataFrame(data=pcor,index=cor.index,columns=cor.columns)
    p_df=pd.DataFrame(data=p,index=cor.index,columns=cor.columns)
    return pcor_df,p_df

#有意な偏相関係数リスト
#  df : pandas dataFrame format
#  alpha : max of p-value
def pcorlist(df,alpha=0.05):
    r,p = pcorr(df)
    clist = []
    n = df.columns.size
    j = 0
    while j < n:
        i = 0
        while i < j:
            if( p.iloc[i,j] < alpha ):
                first = r.columns[i]
                second= r.columns[j]
                rvalue= r.iloc[i,j]
                pvalue= p.iloc[i,j]
                dat = { 'first':first, 'second':second, 'r':rvalue, 'p':pvalue }
                clist.append(dat)
            i+=1
        j+=1
    return clist

# 有意な相関係数リストを表示する
def print_clist(clist):
    for item in clist:
        print("r=%.2f, p=%.4f : %-16s <-> %-16s"%(item['r'],item['p'],item['first'],item['second']))

