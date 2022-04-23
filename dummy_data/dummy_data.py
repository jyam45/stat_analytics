import numpy as np
import pandas as pd

#リッカート式アンケートのダミー回答データを生成
#
# Args
#  nitems     : 質問数
#  nsamples   : 回答者数
#  min_likert : リッカート形式回答の最小値
#  max_likert : リッカート形式回答の最大値
#
# Return
#  pandas.DataFrame
#
def make_dummy_likert(nitems,nsamples,min_likert=1,max_likert=7):
    likert_range = max_likert - min_likert + 1
    dummy = np.random.normal((min_likert+max_likert-1)/2,1.4,(nsamples,nitems)).round(0).astype(np.int64) # normal dist.
    dummy = dummy % likert_range + min_likert
    data  = pd.DataFrame(dummy)
    # ラベル生成
    item_labels = []
    i = 0
    while i < nitems :
        label = 'Q' + str(i+1)
        item_labels.append(label)
        i+=1    
    data.columns = item_labels
    
    return data #pd.DataFrameで返却


