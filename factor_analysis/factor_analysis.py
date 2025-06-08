import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import semopy as sem
import japanize_matplotlib
 
from factor_analyzer import FactorAnalyzer
 
class ExploratoryFactorAnalysis:
     
    def __init__(self,
                 likert=(1,7),tol_loading=0.5,tol_alpha=0.7,tol_scree=0.5,rotation='promax',
                 ddof=False,dropping=False,cfa=False,cfa2=False,bifactor=False,cfa2l=False,
                 method='minres',bounds=(0.005,1),impute='median',tol_95ci=0.95):
        self.likert_          = likert         # Likert's scale for reversing (default:(1,7))
        self.tol_loading_     = tol_loading    # Cutoff value for factor loadings (default:0.5)
        self.tol_alpha_       = tol_alpha      # Cutoff value for Cronbach's alpha (default:0.7)
        self.tol_scree_       = tol_scree      # Cutoff value in Scree's plot to detect max number of factors (default:0.5)
        self.tol_95ci_        = tol_95ci       # Cutoff percentage for confidence intervals (default:0.95)
        self.rotation_        = rotation       # [None|'varimax'|'promax'|'oblimin'|'oblimax'|'quartimin'|'quatimax'|'eqamax']
        self.ddof_            = ddof           # [True|False](default:False) 不偏分散
        self.dropping_        = dropping       # [True|False](defualt:False) switch for dropping items.
        self.cfa_             = cfa            # [True|False](defualt:False) do CFA at the end of analyze()
        self.cfa2_            = cfa2           # [True|False](defualt:False) do 2nd-order CFA at the end of analyze()
        self.bfct_            = bifactor       # [True|False](defualt:False) do Bifactor CFA at the end of analyze()
        self.lyr2_            = cfa2l          # [True|False](defualt:False) do 2nd layered CFA at the end of analyze()
        self.method_          = method         # ['minres'|'ml'|'principal']
        self.bounds_          = bounds         # L-BFGS-M bounds
        self.impute_          = impute         # ['median'|'drop'|'mean']
        self.nfactors_        = 0              # number of factors
        self.max_factors_     = 0              # maximum number of factors
        self.drop_list_       = None           # Drop Item's column names as pd.Index type
        self.scree_eigvals_   = None           # Eigen values for scree plot
        self.factors_         = None           # factor values
        self.alphas_          = None           # Cronbach's alpha
        self.omegas_          = None           # McDonald's omega
        self.loadings_        = None           # The factor loading matrix
        self.corr_            = None           # The original correlation matrix
        self.rotation_matrix_ = None           # The rotation matrix
        self.structure_       = None           # The structure loading matrix for promax
        self.factor_corr_     = None           # The factor correlations matrix (psi)
        self.communalities_   = None           # 
        self.eigenvalues_     = None
        self.factor_variance_ = None
        self.uniquenesses_    = None
        self.sorted_loadings_ = None
        self.itcorr_          = None           # I-T correlations
        self.citcorr_         = None           # Cumulative I-T correlations
        self.rhos_            = None           # Split-half rho
        self.cis_             = None           # 95% confidence interval
        self.cfa_model_       = None           # CFA model text
        self.cfa_sem_model_   = None           # SEM model by semopy
        self.cfa_sem_result_  = None           # Results of SEM
        self.cfa_sem_inspect_ = None           #
        self.cfa_sem_gfi_     = None           # Good Fit Indexies
        self.cfa2_model_      = None           # CFA model text
        self.cfa2_sem_model_  = None           # SEM model by semopy
        self.cfa2_sem_result_ = None           # Results of SEM
        self.cfa2_sem_inspect_= None           #
        self.cfa2_sem_gfi_    = None           # Good Fit Indexies
        self.bfct_model_      = None           # CFA model text
        self.bfct_sem_model_  = None           # SEM model by semopy
        self.bfct_sem_result_ = None           # Results of SEM
        self.bfct_sem_inspect_= None           #
        self.bfct_sem_gfi_    = None           # Good Fit Indexies
        self.lyr2_model_      = None           # CFA model text
        self.lyr2_sem_model_  = None           # SEM model by semopy
        self.lyr2_sem_result_ = None           # Results of SEM
        self.lyr2_sem_inspect_= None           #
        self.lyr2_sem_gfi_    = None           # Good Fit Indexies
         
    #クーロンバックα係数
    @staticmethod
    def cronbach_alpha(data, ddof=False):
        #ddof=True:不偏分散、False:標本分散
        M = data.shape[1]
         
        if M > 1 :
            x = data.sum(axis=1) # 行方向への和
             
            #分散
            sigma2_x = x.var(ddof=ddof)    
            sigma2_i = 0e0
            for i in range(M):
                sigma2_i += data.iloc[:,i].var(ddof=ddof)
         
            cronbach_alpha = M / (M-1) * ( 1 - sigma2_i/sigma2_x )
        else:
            cronbach_alpha = float(M)
             
        return cronbach_alpha

    #オメガ係数（McDonald, 1999）
    @staticmethod
    def mcdonald_omega(data, loads, ddof=False):
        #ddof=True:不偏分散、False:標本分散
        M = data.shape[1]
         
        if M > 1 :
            x = data.sum(axis=1) # 行方向への和
             
            #分散
            sigma2_x = x.var(ddof=ddof) 
            lambda_i = 0e0 
            for i in range(M):
                lambda_i += abs(loads[i])
         
            mcdonald_omega = (lambda_i*lambda_i)/sigma2_x
        else:
            mcdonald_omega = float(M)
             
        return mcdonald_omega
    
    #信頼区間 
    @staticmethod
    def confidence_interval(data, percent=0.95):
        #ddof=True:不偏分散、False:標本分散。基本的に不偏分散を用いる。
        n = data.shape[0] # サンプル数
        q = 1 - ( 1 - percent ) /2
        t = st.t.ppf(q=q,df=n-1) # t分布
        x = data.mean(axis=1) # 行方向への平均
        x_mean = x.mean()
        x_std  = x.std(ddof=True) # 標準偏差
        margin = t * ( x_std / np.sqrt(n) )
        return ( x_mean - margin, x_mean + margin )

    # I-T相関
    # https://bellcurve.jp/statistics/glossary/7396.html
    @staticmethod
    def item_total_corr(data):
        m = len(data.columns)
        x = data.copy() # deep copy
        x = x.join(pd.DataFrame(data.sum(axis=1),columns=["I-T corr."])) # 行方向の和を追加する
        c = x.corr()
        z = pd.DataFrame(c.loc[:,"I-T corr."].iloc[:m,])
        z = z.sort_values("I-T corr.",ascending=False)
        return z

    # 累積I-T相関
    # https://bellcurve.jp/statistics/glossary/7396.html
    @staticmethod
    def cumu_item_total_corr(data):
        m = len(data.columns)
        x = data.copy() # deep copy
        x = x.join(pd.DataFrame(data.sum(axis=1),columns=["I-T corr."])) # 行方向の和を追加する
        c = x.corr()
        z = pd.DataFrame(c.loc[:,"I-T corr."].iloc[:m,])
        z = z.sort_values("I-T corr.",ascending=False)
        y = pd.DataFrame(x.loc[:,z.index[0]])
        for k in range(len(z.index)-1):
            y = y.join(pd.DataFrame(y.iloc[:,k]+x.loc[:,z.index[k+1]],columns=[z.index[k+1]]))
        y = y.join(pd.DataFrame(x.loc[:,"I-T corr."]))
        r = y.corr()
        w = pd.DataFrame(r.loc[:,"I-T corr."].iloc[:m,])
        w.columns=["Cumu I-T corr."]
        return z, w
     
    # 折半法(split-half method)
    # https://bellcurve.jp/statistics/glossary/7447.html
    @staticmethod
    def split_half_rho(data):
        x = {}
        x["even"] = data.iloc[:,0::2].sum(axis=1)
        x["odd" ] = data.iloc[:,1::2].sum(axis=1)
        z = pd.DataFrame(data=x)
        c = z.corr()
        r = c.loc["even","odd"]
        rho = 2*r /(1+r)
        return rho
     
    #標準化
    #https://note.nkmk.me/python-list-ndarray-dataframe-normalize-standardize/
    @staticmethod
    def standardize(data,ddof=False):
        stddat = ( data - data.mean() ) / data.std(ddof=ddof)
        stddat.index = data.index
        stddat.columns = data.columns
        return stddat

    #逆転（内部関数）
    def __reverse(self,value):
        #if self.rtype is "likert" :
        return (self.likert_[0]+self.likert_[1]) - value

    #因子負荷量から項目データを選択する（内部関数）
    def __select_items_by_loading(self,loading,data): # dataは標準化ずみなこと
        # 格納用辞書：pandas.DataFrameに変換するときに、列データとするため辞書に
        item_dict = {} 
        # 因子負荷量の絶対値が閾値より大きい質問を取り出す
        for i in range(len(loading)):
            if loading[i] >= self.tol_loading_ :
                item_dict[data.columns[i]] = data.values[:,i]
            elif loading[i] <= -self.tol_loading_ :
                #item_dict[data.columns[i]] = self.__reverse(data.values[:,i])
                item_dict[data.columns[i]] = -data.values[:,i] # data is standardized
        return pd.DataFrame(item_dict)

    #因子負荷量から有効な因子負荷量を選択する（内部関数）
    def __select_loads_by_loading(self,loading,data): # dataは標準化ずみなこと
        # 格納用辞書：pandas.DataFrameに変換するときに、列データとするため辞書に
        load_list = []
        # 因子負荷量の絶対値が閾値より大きい質問を取り出す
        for i in range(len(loading)):
            if loading[i] >= self.tol_loading_ :
                load_list.append(loading[i])
            elif loading[i] <= -self.tol_loading_ :
                load_list.append(-loading[i])
        return load_list

    #１因子分析による因子負荷量を計算する（内部関数）
    def __calc_single_factor_loads(self,items): # itemsは部分データ
        # 格納用辞書：pandas.DataFrameに変換するときに、列データとするため辞書に
        load_list = []
        #１因子分析
        fa = FactorAnalyzer(1,rotation=self.rotation_,method=self.method_,
                            bounds=self.bounds_,impute=self.impute_)
        #print(data)
        #print(items)
        fa.fit(items) # 因子分析（特異値分解SVD）
        # 因子負荷量の絶対値が閾値より大きい質問を取り出す
        for i in range(len(fa.loadings_)):
            ld = fa.loadings_[i]
            if ld >= self.tol_loading_ :
                load_list.append(ld)
            elif ld <= -self.tol_loading_ :
                load_list.append(-ld)
        return load_list

        
    #因子負荷量からクロンバックαを計算する（内部関数）
    def __loading_to_alpha(self,loading,data):
        items = self.__select_items_by_loading(loading,data)
        #クロンバックαの計算
        alpha = 0e0
        if len(items.columns) > 1 : # １質問の場合は、共通因子はなかったと見なす
            alpha = self.cronbach_alpha(items,ddof=self.ddof_)
        #print(item_dict.keys()," alpha="+str(alpha))
        return alpha

    def __loadings_to_alpha(self,loadings,data):
        nfactors = len(loadings.columns)
        factnames= loadings.columns
        alphas = pd.DataFrame(np.zeros(nfactors).reshape(1,nfactors),index=["alpha"],columns=factnames)
        for i in range(nfactors):
            items = self.__select_items_by_loading(loadings.iloc[:,i],data)
            #クロンバックαの計算
            alpha = 0e0
            if len(items.columns) > 1 : # １質問の場合は、共通因子はなかったと見なす
                alpha = self.cronbach_alpha(items,ddof=self.ddof_)
            alphas.iloc[0,i] = alpha
            #print(item_dict.keys()," alpha="+str(alpha))
        return alphas

    #因子負荷量からマクドナルドωを計算する（内部関数）
    def __loading_to_omega(self,loading,data):
        items = self.__select_items_by_loading(loading,data)
        #loads = self.__select_loads_by_loading(loading,data)
        loads = self.__calc_single_factor_loads(items)
        #マクドナルドωの計算
        omega = 0e0
        if len(items.columns) > 1 : # １質問の場合は、共通因子はなかったと見なす
            omega = self.mcdonald_omega(items,loads,ddof=self.ddof_)
        #print(item_dict.keys()," alpha="+str(alpha))
        return omega

    def __loadings_to_omega(self,loadings,data):
        nfactors = len(loadings.columns)
        factnames= loadings.columns
        omegas = pd.DataFrame(np.zeros(nfactors).reshape(1,nfactors),index=["omega"],columns=factnames)
        for i in range(nfactors):
            items = self.__select_items_by_loading(loadings.iloc[:,i],data)
            #loads = self.__select_loads_by_loading(loadings.iloc[:,i],data)
            loads = self.__calc_single_factor_loads(items)
            #マクドナルドωの計算
            omega = 0e0
            if len(items.columns) > 1 : # １質問の場合は、共通因子はなかったと見なす
                omega = self.mcdonald_omega(items,loads,ddof=self.ddof_)
            omegas.iloc[0,i] = omega
            #print(item_dict.keys()," omega="+str(omega))
        return omegas

    #信頼区間を計算する（内部関数）
    def __loading_to_ci(self,loading,data):
        items = self.__select_items_by_loading(loading,data)
        return self.confidence_interval(items,self.tol_95ci_)

    def __loadings_to_ci(self,loadings,data):
        nfactors = len(loadings.columns)
        factnames= loadings.columns
        indexname= format(int(self.tol_95ci_*100), "2d") + "%CI"
        cis = pd.DataFrame(np.zeros(2*nfactors).reshape(2,nfactors),index=[indexname+"_min",indexname+"_max"],columns=factnames)
        for i in range(nfactors):
            items = self.__select_items_by_loading(loadings.iloc[:,i],data)
            # 95%信頼区間の計算
            ci = self.confidence_interval(items,self.tol_95ci_)
            #print("CI=",ci)
            cis.iloc[0,i] = ci[0]
            cis.iloc[1,i] = ci[1]
            #cis.iloc[0,i] = self.confidence_interval(items,self.tol_95ci_)
        return cis


    #I-T相関を計算する
    def __loading_to_itcorr(self,loading,data):
        items = self.__select_items_by_loading(loading,data)
        itcorr= self.item_total_corr(items)
        return itcorr

    def __loadings_to_itcorr(self,loadings,data):
        nfactors = len(loadings.columns)
        factnames= loadings.columns
        itcorrs  = pd.DataFrame()
        #itcorrs  = []
        for i in range(nfactors):
            items = self.__select_items_by_loading(loadings.iloc[:,i],data)
            itcorr= self.item_total_corr(items)
            itcorr.columns = [factnames[i]]
            #itcorrs = itcorrs.append(itcorr) # pandas2で削除された
            itcorrs = pd.concat([itcorrs,itcorr],ignore_index=False) # pandas2で削除された
        return itcorrs
        #return pd.DataFrame(itcorrs)

    #I-T相関と累積I-T相関を計算する
    #  itcorr : I-T相関
    #  citcorr: 累積I-T相関
    def __loading_to_citcorr(self,loading,data):
        items = self.__select_items_by_loading(loading,data)
        itcorr,citcorr= self.cumu_item_total_corr(items)
        return itcorr,citcorr

    def __loadings_to_citcorr(self,loadings,data):
        nfactors = len(loadings.columns)
        factnames= loadings.columns
        itcorrs  = pd.DataFrame()
        citcorrs = pd.DataFrame()
        #itcorrs  = []
        for i in range(nfactors):
            items = self.__select_items_by_loading(loadings.iloc[:,i],data)
            itcorr, citcorr= self.cumu_item_total_corr(items)
            itcorr.columns = [factnames[i]]
            citcorr.columns= [factnames[i]]
            #itcorrs = itcorrs.append(itcorr) # pandas2で削除された
            itcorrs = pd.concat([itcorrs,itcorr],ignore_index=False) # pandas2で削除された
            citcorrs= pd.concat([citcorrs,citcorr],ignore_index=False) # pandas2で削除された
        return itcorrs,citcorrs
        #return pd.DataFrame(itcorrs)
     
    #折半法を計算する
    def __loading_to_rho(self,loading,data):
        items = self.__select_items_by_loading(loading,data)
        rho   = self.split_half_rho(items)
        return rho

    def __loadings_to_rho(self,loadings,data):
        nfactors = len(loadings.columns)
        factnames= loadings.columns
        rhos     = pd.DataFrame(np.zeros(nfactors).reshape(1,nfactors),index=["rho"],columns=factnames)
        for i in range(nfactors):
            items = self.__select_items_by_loading(loadings.iloc[:,i],data)
            rho   = self.split_half_rho(items)
            rhos.iloc[0,i] = rho
        return rhos

    #因子間相関を計算する 
    def __loadings_to_corr(self,loadings,data):
        m  = len(data.index)
        nf = len(loadings.columns)
        fnames = loadings.columns
        x = pd.DataFrame(np.zeros(m*nf).reshape(m,nf),index=data.index,columns=fnames)
        for i in range(nf):
            picks = self.__select_items_by_loading(loadings.iloc[:,i],data)
            x[fnames[i]] = picks.mean(axis=1)
        return x.corr()

    #複数因子に含まれる項目を見つける
    def __find_drop_items(self,loadings):
        #print(loadings)
        a = (abs(loadings) >= self.tol_loading_).sum(axis=1) # 複数因子に含まれる項目確認 
        df = pd.DataFrame(a,index=loadings.index,columns=["count"])
        #print(df)
        x = df[ df["count"] > 1 ]
        return x.index

    #ソート済因子負荷量
    @staticmethod
    def sort_loadings(loadings,tol=0.4):
        i=1
        n=len(loadings.columns)
        #print(n)
        names=[]
        while i <= n:
            name="A"+str(i)
            names.append(name)
            i+=1
        abs_loadings = loadings.abs()
        abs_loadings.columns = names
        tmp_loadings = loadings.copy() # deep copy
        tmp_loadings = tmp_loadings.join(abs_loadings)
 
        #print(tld)
        results=pd.DataFrame()
        #results=[]
        ilabel=n
        for label in names:
            x = tmp_loadings.sort_values(label,ascending=False)
            #print(label)
            #print(x)
            # explore
            j=0
            while j < len(x.index) and x.iloc[j,ilabel] >= tol:
                j += 1
            #print(x.iloc[0:j,0:n])
            #results = results.append(x.iloc[0:j,0:n]) # pandas2でappend()削除->concat()推奨に
            results = pd.concat([results,pd.DataFrame(x.iloc[0:j,0:n])],ignore_index=False) 
            tmp_loadings = x.iloc[j:,:]
            ilabel += 1
        #print(results)
        return results
        #return pd.DataFrame(results)
     
    #因子数推定
    def explore( self, data ):
     
        #標準化
        stddat = self.standardize(data,ddof=self.ddof_)
     
        #最大因子数推定
        ev = pd.DataFrame(np.linalg.eigvals(stddat.corr()))
        max_factors = 0
        for e in ev[0]:
            if e > self.tol_scree_ :
                max_factors += 1
        print('predicted max_factors = '+str(max_factors))
     
        #因子数探索ループ
        nfactors = max_factors
        drop_list= pd.Index([],dtype="object") if self.dropping_  else None
        while nfactors > 0 :
         
            print('try nfactors = '+str(nfactors))
             
            #因子分析
            fa = FactorAnalyzer(nfactors,rotation=self.rotation_,method=self.method_,
                                bounds=self.bounds_,impute=self.impute_)
            fa.fit(stddat) # 因子分析（特異値分解SVD）
     
            #因子負荷量(因子×質問）
            #loadings = pd.DataFrame(fa.components_.T)
            loadings = pd.DataFrame(fa.loadings_,index=stddat.columns)
             
            #信頼性＆妥当性（内的一貫性）のチェック
            ifactor = nfactors - 1
            while ifactor >= 0:
                alpha  = self.__loading_to_alpha(loadings.iloc[:,ifactor],stddat)
                if alpha < self.tol_alpha_ : # 閾値未満なら一貫性がないため、因子数が妥当ではない。
                    break
                ifactor -= 1
 
            #探索終了条件：全因子の妥当性が確認された場合、探索を打ち切る
            if ifactor < 0 : 
                if self.dropping_ :
                    drop_items = self.__find_drop_items(loadings) #重複項目の探索
                    #print(stddat)
                    if len(drop_items) > 0 :
                        print("Dropped Items : ",drop_items)
                        stddat = stddat.drop(drop_items,axis=1)
                        nfactors = max_factors+1 # 再探索する
                        drop_list = drop_list.union(drop_items)
                    else:
                        print('found optimum nfactors = '+str(nfactors))
                        break
                else:
                    print('found optimum nfactors = '+str(nfactors))
                    break
         
            #探索継続条件：因子が確認できなかった場合、因子数を減らして次へ。
            nfactors -= 1
         
        #推定因子数（もし0なら失敗）
        if nfactors > 0:
            print("Predicted nfactors = "+str(nfactors))
        else:
            print("Prediction error : No factors in this data, because nfactors = 0")
         
        #インスタンス変数に格納
        self.max_factors_    = max_factors
        self.nfactors_       = nfactors
        self.scree_eigvals_  = ev
        self.drop_list_      = drop_list
         
        return nfactors, drop_list # 成功 nfactors>0, 失敗 nfactors=0
     
    def scree_plot(self):
        # 基準線(固有値1)
        ev_1 = np.ones(len(self.scree_eigvals_))
 
        # 変数
        plt.plot(self.scree_eigvals_, 's-')   # 主成分分析による固有値
        plt.plot(ev_1, 's-') # ダミーデータ
 
        # 軸名
        plt.xlabel("因子の数")
        plt.ylabel("固有値")
 
        plt.grid()
        plt.show()
 
    #因子分析の解説
    #
    def analyze(self,data,nfactors,drop_list=None):
 
        #エラー処理
        if nfactors < 1 :
            raise ValueError('NFACTORS in analyze() must be more than zero!')

        #仮因子名生成 "Fn"
        factnames = [] # 因子名配列
        i=0
        while i < nfactors: #因子ラベル生成
            factname = 'F' + str(i+1)
            factnames.append(factname)
            i+=1
             
        #ドロップ項目削除（dataは参照なので変更しない）
        dropped = data.drop(drop_list,axis=1) if drop_list is not None  else data

        #標準化
        stddat = self.standardize(data,ddof=self.ddof_)
 
        #因子分析（MINRES法を使用）
        fa = FactorAnalyzer(nfactors,rotation=self.rotation_,method=self.method_,
                            bounds=self.bounds_,impute=self.impute_)
        fa.fit(stddat)

        #重複項目の削除 & 再分析
        #dropped = data # 項目削除された非標準化データを使う場合に使用すること。これ以降dataは使用してはならない。
        if self.dropping_ :
            drop_items = self.__find_drop_items(pd.DataFrame(fa.loadings_,index=stddat.columns,columns=factnames))
            while len(drop_items) > 0 :
                print("Dropped Items : ",drop_items)
                stddat = stddat.drop(drop_items,axis=1)
                dropped= dropped.drop(drop_items,axis=1)
                fa.fit(stddat)
                drop_items = self.__find_drop_items(pd.DataFrame(fa.loadings_,index=stddat.columns,columns=factnames))
            
        #因子負荷量（質問×因子）
        self.loadings_ = pd.DataFrame(fa.loadings_,index=stddat.columns,columns=factnames)
        self.sorted_loadings_ = self.sort_loadings(self.loadings_,tol=self.tol_loading_)
 
        #内的一貫性
        #self.alphas_ = pd.DataFrame(np.zeros(nfactors).reshape(1,nfactors),index=["alpha"],columns=factnames)
        #for i in range(nfactors):
        #    alpha = self.__loading_to_alpha(self.loadings_.iloc[:,i],stddat)#stddat?
        #    self.alphas_.iloc[0,i] = alpha
        self.alphas_ = self.__loadings_to_alpha(self.loadings_,stddat)
        self.omegas_ = self.__loadings_to_omega(self.loadings_,stddat)
         
        #I-T相関と累積I-T相関
        #self.itcorr_ = pd.DataFrame()
        #for i in range(nfactors):
        #    itcorr = self.__loading_to_itcorr(self.loadings_.iloc[:,i],stddat)#stddat?
        #    itcorr.columns = [factnames[i]]
        #    self.itcorr_ = self.itcorr_.append(itcorr)
        #self.itcorr_ = self.__loadings_to_itcorr(self.loadings_,stddat)
        self.itcorr_,self.citcorr_ = self.__loadings_to_citcorr(self.loadings_,stddat)
         
        #split-half
        #self.rhos_ = pd.DataFrame(np.zeros(nfactors).reshape(1,nfactors),index=["rho"],columns=factnames)
        #for i in range(nfactors):
        #    rho = self.__loading_to_rho(self.loadings_.iloc[:,i],stddat)#stddat?
        #    self.rhos_.iloc[0,i] = rho
        self.rhos_ = self.__loadings_to_rho(self.loadings_,stddat)

        #信頼区間
        self.cis_  = self.__loadings_to_ci(self.loadings_,dropped) #信頼区間の計算は非標準化得点を使う
        #self.cis_  = self.__loadings_to_ci(self.loadings_,stddat) #標準化得点を使う場合はこちら
 
        #因子得点
        self.factors_ = pd.DataFrame(fa.transform(stddat),index=stddat.index,columns=factnames)  # 因子得点に変換

        #共通性（共通分散の値のこと、複数の観測変数に影響する度合い）
        x = fa.get_communalities()
        if x is not None :
            self.communalities_ = pd.DataFrame(x,index=stddat.columns,columns=['comm.'])
            #print(self.communalities_)
         
        #オリジナル相関の固有値、因子相関の固有値
        x = fa.get_eigenvalues()
        if x is not None :
            self.eigenvalues_ = pd.DataFrame(x,index = ['original','factor'],columns = stddat.columns)
            #print(self.eigenvalues_)
         
        #因子寄与、因子寄与率、累積因子寄与率
        x = fa.get_factor_variance()
        if x is not None :
            self.factor_variance_ = pd.DataFrame(x,index=['var.','prop.','cumu.'],columns = factnames)
            #print(self.factor_variance_)
 
        #独自性（独自分散の値のこと、単数の観測変数に影響する度合い）
        x = fa.get_uniquenesses()
        if x is not None :
            self.uniquenesses_ = pd.DataFrame(x,index = stddat.columns,columns = ['uniq.'])
            #print(self.uniquenesses_)
 
        #相関行列（項目間の相関）
        self.corr_ = pd.DataFrame(fa.corr_,index=stddat.columns,columns=stddat.columns)
        #print(self.corr_)
         
        #回転行列（斜交回転の場合だけ計算され、軸の回転を表す）
        if hasattr(fa,'rotation_matrix_') :
            self.rotation_matrix_ = pd.DataFrame(fa.rotation_matrix_,index=factnames,columns=factnames)
            #print(self.rotation_matrix_)
        else:
            print("'fa' does not have a menber 'rotation_matrix_'")
         
        #構造行列（promaxの場合だけ計算される）
        if hasattr(fa,'structure_'):
            self.structure_ = pd.DataFrame(fa.structure_,index=stddat.columns,columns=factnames)
            #print(self.structure_)
        else:
            print("'fa' does not have a menber 'structure_'")
         
        #因子相関行列（斜交回転の場合だけ計算される、promaxで計算されないのはバグ？）
        if hasattr(fa,'psi_'):
            self.factor_corr_ = pd.DataFrame(fa.psi_)
            #print(self.factor_corr_)
        else:
            print("Computed factor's correlations instead because 'fa' does not have a member 'psi_'")
            self.factor_corr_ = self.__loadings_to_corr(self.loadings_,stddat)

        #１次のCFAを計算する
        if self.cfa_:
           self.cfa_model_      = self.cfa_model()
           self.cfa_sem_model_  = sem.Model(self.cfa_model_)
           self.cfa_sem_result_ = self.cfa_sem_model_.fit(stddat)
           self.cfa_sem_inspect_= self.cfa_sem_model_.inspect(std_est=True)
           self.cfa_sem_gfi_    = sem.calc_stats(self.cfa_sem_model_)

        #２次のCFAを計算する
        if self.cfa2_ and nfactors > 1 :
           self.cfa2_model_      = self.cfa_model(model_type="order2")
           self.cfa2_sem_model_  = sem.Model(self.cfa2_model_)
           self.cfa2_sem_result_ = self.cfa2_sem_model_.fit(stddat)
           self.cfa2_sem_inspect_= self.cfa2_sem_model_.inspect(std_est=True)
           self.cfa2_sem_gfi_    = sem.calc_stats(self.cfa2_sem_model_)

        #Bifactor CFAを計算する
        if self.bfct_ and nfactors > 1 :
           self.bfct_model_      = self.cfa_model(model_type="bifactor")
           self.bfct_sem_model_  = sem.Model(self.bfct_model_)
           self.bfct_sem_result_ = self.bfct_sem_model_.fit(stddat)
           self.bfct_sem_inspect_= self.bfct_sem_model_.inspect(std_est=True)
           self.bfct_sem_gfi_    = sem.calc_stats(self.bfct_sem_model_)

        #２層CFAを計算する
        if self.lyr2_ and nfactors > 1 :
           self.lyr2_model_      = self.cfa_model(model_type="layer2")
           self.lyr2_sem_model_  = sem.Model(self.lyr2_model_)
           self.lyr2_sem_result_ = self.lyr2_sem_model_.fit(stddat)
           self.lyr2_sem_inspect_= self.lyr2_sem_model_.inspect(std_est=True)
           self.lyr2_sem_gfi_    = sem.calc_stats(self.lyr2_sem_model_)
         
        return self.factors_

    # 簡易実行関数
    def find(self,data):
        nf,dl = self.explore(data)
        return self.analyze(data,nf,drop_list=dl)

    # CFAのための構造方程式を計算する
    def cfa_model(self,formating="lavaan",model_type="order1",gfactor="GF"):
        model = ""
        if formating == "lavaan":
           left = "~"
           right= "=~"
           both = "~~"
        elif formating == "sem":
           left = "<-"
           right= "->"
           both = "<>"
        else:
           print(f"cfa_model: irregal option formating={formating}. ['lavaan','sem']")
        factors = self.sorted_loadings_.columns
        items   = self.sorted_loadings_.index
        #潜在変数の定義
        for factor in factors:
            for item in items:
                #print(item,factor)
                loading = self.sorted_loadings_.loc[item,factor]
                if abs(loading) > self.tol_loading_:
                    model += f"{factor} {right} {item}\n"
            #model += f"{factor} {both} 1*{factor}\n"
        #CFAモデル
        if model_type == "order1":
            # 通常のCFAモデル
            nfactors = len(factors)
            for j in range(1,nfactors):
                jfactor = factors[j]
                for i in range(0,j):
                    ifactor = factors[i]
                    model += f"{ifactor} {both} {jfactor}\n"
        elif model_type == "order2":
            # ２次因子モデル
            nfactors = len(factors)
            for j in range(0,nfactors):
                jfactor = factors[j]
                model += f"{gfactor} {right} {jfactor}\n"
            #model += f"DEFINE(latent) {gfactor}\n"
        elif model_type == "bifactor":
            # bifactorモデル
            for item in items:
                model += f"{gfactor} {right} {item}\n"
            # 潜在因子の独立性
            for factor in factors:
                model += f"{gfactor} {both} 0*{factor}\n"
            nfactors = len(factors)
            for j in range(1,nfactors):
                jfactor = factors[j]
                for i in range(0,j):
                    ifactor = factors[i]
                    model += f"{ifactor} {both} 0*{jfactor}\n"
            #model += f"DEFINE(latent) {gfactor}\n"
            #model += f"{gfactor} {both} 1*{gfactor}\n"
        elif model_type == "layer2":
            # 階層因子モデル
            nfactors = len(factors)
            for j in range(0,nfactors):
                jfactor = factors[j]
                model += f"{gfactor} {right} {jfactor}\n"
            for item in items:
                model += f"{gfactor} {right} {item}\n"
            #model += f"DEFINE(latent) {gfactor}\n"
        else:
            print("cfa_model: invalide option model_type={model_type}. ['order1','order2','bifactor','layer2']")
        return model
 
    def __str__(self):
        x  = "\n"
        x += "Likert's scale : " + str(self.likert_) + "\n"
        x += "Cutoff Loading : " + str(self.tol_loading_) + "\n"
        x += "Cutoff Alpha   : " + str(self.tol_alpha_) + "\n"
        x += "Cutoff CI      : " + str(self.tol_95ci_) + "\n"
        x += "Rotation       : " + self.rotation_ + "\n"
        x += "Num. of Factors: " + str(self.nfactors_) + "\n"
        x += "\n"
        if self.loadings_ is not None:
            pd.set_option('display.max_columns', len(self.loadings_.columns))
            pd.set_option('display.max_rows', len(self.loadings_.index))       
        if self.sorted_loadings_ is not None:
            x += "Loadings:\n" + str(self.sorted_loadings_) + "\n\n"
        if self.alphas_ is not None:
            x += "Cronbach's alpha:\n" + str(self.alphas_) + "\n\n"
        if self.omegas_ is not None:
            x += "McDonald's omega:\n" + str(self.omegas_) + "\n\n"
        if self.rhos_ is not None:
            x += "Split-half rho:\n" + str(self.rhos_) + "\n\n"
        if self.cis_ is not None:
            x += "Confidence Intervals:\n" + str(self.cis_) + "\n\n"
        if self.scree_eigvals_ is not None:
            pd.set_option('display.max_rows',len(self.scree_eigvals_.index))
            x += "Scree eignvalues:\n" + str(self.scree_eigvals_) + "\n\n"
        if self.itcorr_ is not None:
            x += "I-T Correlations:\n" + str(self.itcorr_) + "\n\n"
        if self.citcorr_ is not None:
            x += "Cumulative I-T Correlations:\n" + str(self.citcorr_) + "\n\n"
        if self.rotation_matrix_ is not None:
            x += "Rotation Matrix:\n" + str(self.rotation_matrix_) + "\n\n"
        if self.factor_corr_ is not None:
            x += "Factor Correlations:\n" + str(self.factor_corr_) + "\n\n"
        if self.factor_variance_ is not None:
            x += "Contributions:\n" + str(self.factor_variance_) + "\n\n"
        if self.communalities_ is not None:
            x += "Communalities:\n" + str(self.communalities_) + "\n\n"
        if self.uniquenesses_ is not None:
            x += "Uniquenesses:\n" + str(self.uniquenesses_) + "\n\n"
        if self.cfa_model_ is not None:
            x += "CFA Model:\n" + str(self.cfa_model_) + "\n\n"
        if self.cfa_sem_result_ is not None:
            x += "CFA Information:\n" + str(self.cfa_sem_result_) + "\n\n"
        if self.cfa_sem_inspect_ is not None:
            pd.set_option('display.max_columns',len(self.cfa_sem_inspect_.columns))
            pd.set_option('display.max_rows',len(self.cfa_sem_inspect_.index))
            x += "CFA Result:\n" + str(self.cfa_sem_inspect_) + "\n\n"
        if self.cfa_sem_gfi_ is not None:
            pd.set_option('display.max_columns',len(self.cfa_sem_gfi_.index))
            pd.set_option('display.max_rows',len(self.cfa_sem_gfi_.columns))
            x += "CFA Good Fit Index:\n" + str(self.cfa_sem_gfi_.T) + "\n\n"
        if self.cfa2_model_ is not None:
            x += "2nd-order CFA Model:\n" + str(self.cfa2_model_) + "\n\n"
        if self.cfa2_sem_result_ is not None:
            x += "2nd-order CFA Information:\n" + str(self.cfa2_sem_result_) + "\n\n"
        if self.cfa2_sem_inspect_ is not None:
            pd.set_option('display.max_columns',len(self.cfa2_sem_inspect_.columns))
            pd.set_option('display.max_rows',len(self.cfa2_sem_inspect_.index))
            x += "2nd-order CFA Result:\n" + str(self.cfa2_sem_inspect_) + "\n\n"
        if self.cfa2_sem_gfi_ is not None:
            pd.set_option('display.max_columns',len(self.cfa2_sem_gfi_.index))
            pd.set_option('display.max_rows',len(self.cfa2_sem_gfi_.columns))
            x += "2nd-order CFA Good Fit Index:\n" + str(self.cfa2_sem_gfi_.T) + "\n\n"
        if self.bfct_model_ is not None:
            x += "Bifactor CFA Model:\n" + str(self.bfct_model_) + "\n\n"
        if self.bfct_sem_result_ is not None:
            x += "Bifactor CFA Information:\n" + str(self.bfct_sem_result_) + "\n\n"
        if self.bfct_sem_inspect_ is not None:
            pd.set_option('display.max_columns',len(self.bfct_sem_inspect_.columns))
            pd.set_option('display.max_rows',len(self.bfct_sem_inspect_.index))
            x += "Bifactor CFA Result:\n" + str(self.bfct_sem_inspect_) + "\n\n"
        if self.bfct_sem_gfi_ is not None:
            pd.set_option('display.max_columns',len(self.bfct_sem_gfi_.index))
            pd.set_option('display.max_rows',len(self.bfct_sem_gfi_.columns))
            x += "Bifactor CFA Good Fit Index:\n" + str(self.bfct_sem_gfi_.T) + "\n\n"
        if self.lyr2_model_ is not None:
            x += "2nd-layered CFA Model:\n" + str(self.lyr2_model_) + "\n\n"
        if self.lyr2_sem_result_ is not None:
            x += "2nd-layered CFA Information:\n" + str(self.lyr2_sem_result_) + "\n\n"
        if self.lyr2_sem_inspect_ is not None:
            pd.set_option('display.max_columns',len(self.lyr2_sem_inspect_.columns))
            pd.set_option('display.max_rows',len(self.lyr2_sem_inspect_.index))
            x += "2nd-layered CFA Result:\n" + str(self.lyr2_sem_inspect_) + "\n\n"
        if self.lyr2_sem_gfi_ is not None:
            pd.set_option('display.max_columns',len(self.lyr2_sem_gfi_.index))
            pd.set_option('display.max_rows',len(self.lyr2_sem_gfi_.columns))
            x += "2nd-layered CFA Good Fit Index:\n" + str(self.lyr2_sem_gfi_.T) + "\n\n"
         
        return x
