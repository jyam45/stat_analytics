import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
 
from factor_analyzer import FactorAnalyzer
 
class ExploratoryFactorAnalysis:
     
    def __init__(self,
                 likert=(1,7),tol_loading=0.5,tol_alpha=0.7,rotation='promax',
                 ddof=False,method='minres',bounds=(0.005,1),impute='median'):
        self.likert_          = likert         # Likert's scale for reversing (default:(1,7))
        self.tol_loading_     = tol_loading    # Cutoff value for factor loadings (default:0.5)
        self.tol_alpha_       = tol_alpha      # Cutoff value for Cronbach's alpha (default:0.7)
        self.rotation_        = rotation       # [None|'varimax'|'promax'|'oblimin'|'oblimax'|'quartimin'|'quatimax'|'eqamax']
        self.ddof_            = ddof           # (default:False) 不偏分散
        self.method_          = method         # ['minres'|'ml'|'principal']
        self.bounds_          = bounds         # L-BFGS-M bounds
        self.impute_          = impute         # ['median'|'drop'|'mean']
        self.nfactors_        = 0              # number of factors
        self.max_factors_     = 0              # maximum number of factors
        self.scree_eigvals_   = None           # Eigen values for scree plot
        self.factors_         = None           # factor values
        self.alphas_          = None           # Cronbach's alpha
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
        self.itcorr_          = None
        self.rhos_            = None
         
    #クーロンバックα係数
    @staticmethod
    def cronbach_alpha(data, ddof=False):
        #ddof=True:不偏分散、False:標本分散
        M = data.shape[1]
         
        if M > 1 :
            x = data.sum(axis=1) # 行方向への和
             
            #分散
            sigma_x = x.var(ddof=ddof)    
            sigma_i = 0e0
            for i in range(M):
                sigma_i += data.iloc[:,i].var(ddof=ddof)
         
            cronbach_alpha = M / (M-1) * ( 1 - sigma_i/sigma_x )
        else:
            cronbach_alpha = float(M)
             
        return cronbach_alpha
     
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
    def standarize(data,ddof=False):
        stddat = data - data.mean() / data.std(ddof=ddof)
        stddat.index = data.index
        stddat.columns = data.columns
 
        return stddat
     
    #逆転（内部関数）
    def __reverse(self,value):
        #if self.rtype is "likert" :
        return (self.likert_[0]+self.likert_[1]) - value
     
    #因子負荷量から項目データを選択する（内部関数）
    def __select_items_by_loading(self,loading,data):
        # 格納用辞書：pandas.DataFrameに変換するときに、列データとするため辞書に
        item_dict = {} 
        # 因子負荷量の絶対値が閾値より大きい質問を取り出す
        for i in range(len(loading)):
            if loading[i] >= self.tol_loading_ :
                item_dict[data.columns[i]] = data.values[:,i]
            elif loading[i] <= -self.tol_loading_ :
                item_dict[data.columns[i]] = self.__reverse(data.values[:,i])
        return pd.DataFrame(item_dict)
         
    #因子負荷量からクロンバックαを計算する（内部関数）
    def __loading_to_alpha(self,loading,data):
        items = self.__select_items_by_loading(loading,data)
        #クロンバックαの計算
        alpha = 0e0
        if len(items.columns) > 1 : # １質問の場合は、共通因子はなかったと見なす
            alpha = self.cronbach_alpha(items,ddof=self.ddof_)
        #print(item_dict.keys()," alpha="+str(alpha))
        return alpha
 
    #I-T相関を計算する
    def __loading_to_itcorr(self,loading,data):
        items = self.__select_items_by_loading(loading,data)
        itcorr= self.item_total_corr(items)
        return itcorr
     
    #折半法を計算する
    def __loading_to_rho(self,loading,data):
        items = self.__select_items_by_loading(loading,data)
        rho   = self.split_half_rho(items)
        return rho
     
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
            results = results.append(x.iloc[0:j,0:n])
            tmp_loadings = x.iloc[j:,:]
            ilabel += 1
        #print(results)
        return results
     
    #因子数推定
    def explore( self, data ):
     
        #標準化
        stddat = self.standarize(data,ddof=self.ddof_)
     
        #最大因子数推定
        ev = pd.DataFrame(np.linalg.eigvals(stddat.corr()))
        max_factors = 0
        for e in ev[0]:
            if e > 1.0 :
                max_factors += 1
        print('predicted max_factors = '+str(max_factors))
     
        #因子数探索ループ
        nfactors = max_factors
        while nfactors > 0 :
         
            print('try nfactors = '+str(nfactors))
             
            #因子分析
            fa = FactorAnalyzer(nfactors,rotation=self.rotation_,method=self.method_,
                                bounds=self.bounds_,impute=self.impute_)
            fa.fit(stddat) # 因子分析（特異値分解SVD）
     
            #因子負荷量(因子×質問）
            #loadings = pd.DataFrame(fa.components_.T)
            loadings = pd.DataFrame(fa.loadings_)
             
            #信頼性＆妥当性（内的一貫性）のチェック
            ifactor = nfactors - 1
            while ifactor >= 0:
                alpha  = self.__loading_to_alpha(loadings.iloc[:,ifactor],data)
                if alpha < self.tol_alpha_ : # 閾値未満なら一貫性がないため、因子数が妥当ではない。
                    break
                ifactor -= 1
 
            #探索終了条件：全因子の妥当性が確認された場合、探索を打ち切る
            if ifactor < 0 : 
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
         
        return nfactors # 成功 nfactors>0, 失敗 nfactors=0
     
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
    def analyze(self,data,nfactors):
 
        #エラー処理
        if nfactors < 1 :
            raise ValueError('NFACTORS in analyze() must be more than zero!')
             
        #標準化
        stddat = self.standarize(data,ddof=self.ddof_)
 
        #因子分析（MINRES法を使用）
        fa = FactorAnalyzer(nfactors,rotation=self.rotation_,method=self.method_,
                            bounds=self.bounds_,impute=self.impute_)
        fa.fit(stddat)
 
        #仮因子名生成 "Fn"
        factnames = [] # 因子名配列
        i=0
        while i < nfactors: #因子ラベル生成
            factname = 'F' + str(i+1)
            factnames.append(factname)
            i+=1
             
        #因子負荷量（質問×因子）
        self.loadings_ = pd.DataFrame(fa.loadings_,index=data.columns,columns=factnames)
        self.sorted_loadings_ = self.sort_loadings(self.loadings_,tol=self.tol_loading_)
 
        #内的一貫性
        self.alphas_ = pd.DataFrame(np.zeros(nfactors).reshape(1,nfactors),index=["alpha"],columns=factnames)
        for i in range(nfactors):
            alpha = self.__loading_to_alpha(self.loadings_.iloc[:,i],data)
            self.alphas_.iloc[0,i] = alpha
         
        #I-T相関
        self.itcorr_ = pd.DataFrame()
        for i in range(nfactors):
            itcorr = self.__loading_to_itcorr(self.loadings_.iloc[:,i],data)
            itcorr.columns = [factnames[i]]
            self.itcorr_ = self.itcorr_.append(itcorr)
         
        #split-half
        self.rhos_ = pd.DataFrame(np.zeros(nfactors).reshape(1,nfactors),index=["rho"],columns=factnames)
        for i in range(nfactors):
            rho = self.__loading_to_rho(self.loadings_.iloc[:,i],data)
            self.rhos_.iloc[0,i] = rho
 
        #因子得点
        self.factors_ = pd.DataFrame(fa.transform(stddat),index=data.index,columns=factnames)  # 因子得点に変換
 
        #共通性（共通分散の値のこと、複数の観測変数に影響する度合い）
        x = fa.get_communalities()
        if x is not None :
            self.communalities_ = pd.DataFrame(x,index=data.columns,columns=['comm.'])
            #print(self.communalities_)
         
        #オリジナル相関の固有値、因子相関の固有値
        x = fa.get_eigenvalues()
        if x is not None :
            self.eigenvalues_ = pd.DataFrame(x,index = ['original','factor'],columns = data.columns)
            #print(self.eigenvalues_)
         
        #因子寄与、因子寄与率、累積因子寄与率
        x = fa.get_factor_variance()
        if x is not None :
            self.factor_variance_ = pd.DataFrame(x,index=['var.','prop.','cumu.'],columns = factnames)
            #print(self.factor_variance_)
 
        #独自性（独自分散の値のこと、単数の観測変数に影響する度合い）
        x = fa.get_uniquenesses()
        if x is not None :
            self.uniquenesses_ = pd.DataFrame(x,index = data.columns,columns = ['uniq.'])
            #print(self.uniquenesses_)
 
        #相関行列（項目間の相関）
        self.corr_ = pd.DataFrame(fa.corr_,index=data.columns,columns=data.columns)
        #print(self.corr_)
         
        #回転行列（斜交回転の場合だけ計算され、軸の回転を表す）
        if hasattr(fa,'rotation_matrix_') :
            self.rotation_matrix_ = pd.DataFrame(fa.rotation_matrix_,index=factnames,columns=factnames)
            #print(self.rotation_matrix_)
        else:
            print("'fa' does not have a menber 'rotation_matrix_'")
         
        #構造行列（promaxの場合だけ計算される）
        if hasattr(fa,'structure_'):
            self.structure_ = pd.DataFrame(fa.structure_,index=data.columns,columns=factnames)
            #print(self.structure_)
        else:
            print("'fa' does not have a menber 'structure_'")
 
         
        #因子相関行列（斜交回転の場合だけ計算される、promaxで計算されないのはバグ？）
        if hasattr(fa,'psi_'):
            self.factor_corr_ = pd.DataFrame(fa.psi_)
            #print(self.factor_corr_)
        else:
            print("'fa' does not have a menber 'psi_'")
 
         
        return self.factors_
 
    def __str__(self):
        x  = "\n"
        x += "Likert's scale : " + str(self.likert_) + "\n"
        x += "Cutoff Loading : " + str(self.tol_loading_) + "\n"
        x += "Cutoff Alpha   : " + str(self.tol_alpha_) + "\n"
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
        if self.rhos_ is not None:
            x += "Split-half rho:\n" + str(self.rhos_) + "\n\n"
        if self.itcorr_ is not None:
            x += "I-T Correlations:\n" + str(self.itcorr_) + "\n\n"
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
         
        return x
