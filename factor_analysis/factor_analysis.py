import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
 
from factor_analyzer import FactorAnalyzer
 
class ExploratoryFactorAnalysis:
     
    def __init__(self,
                 likert=(1,7),tol_loading=0.5,tol_alpha=0.7,tol_scree=0.5,rotation='promax',
                 ddof=False,method='minres',bounds=(0.005,1),impute='median'):
        self.likert_          = likert         # Likert's scale for reversing (default:(1,7))
        self.tol_loading_     = tol_loading    # Cutoff value for factor loadings (default:0.5)
        self.tol_alpha_       = tol_alpha      # Cutoff value for Cronbach's alpha (default:0.7)
	self.tol_scree_       = tol_scree      # Cutoff value in Scree's plot to detect max number of factors (default:0.5)
        self.rotation_        = rotation       # [None|'varimax'|'promax'|'oblimin'|'oblimax'|'quartimin'|'quatimax'|'eqamax']
        self.ddof_            = ddof           # (default:False) δΈεεζ£
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
         
    #γ―γΌγ­γ³γγγ―Ξ±δΏζ°
    @staticmethod
    def cronbach_alpha(data, ddof=False):
        #ddof=True:δΈεεζ£γFalse:ζ¨ζ¬εζ£
        M = data.shape[1]
         
        if M > 1 :
            x = data.sum(axis=1) # θ‘ζΉεγΈγ?ε
             
            #εζ£
            sigma_x = x.var(ddof=ddof)    
            sigma_i = 0e0
            for i in range(M):
                sigma_i += data.iloc[:,i].var(ddof=ddof)
         
            cronbach_alpha = M / (M-1) * ( 1 - sigma_i/sigma_x )
        else:
            cronbach_alpha = float(M)
             
        return cronbach_alpha
     
    # I-TηΈι’
    # https://bellcurve.jp/statistics/glossary/7396.html
    @staticmethod
    def item_total_corr(data):
        m = len(data.columns)
        x = data.copy() # deep copy
        x = x.join(pd.DataFrame(data.sum(axis=1),columns=["I-T corr."])) # θ‘ζΉεγ?εγθΏ½ε γγ
        c = x.corr()
        z = pd.DataFrame(c.loc[:,"I-T corr."].iloc[:m,])
        z = z.sort_values("I-T corr.",ascending=False)
        return z
     
    # ζεζ³(split-half method)
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
     
    #ζ¨ζΊε
    #https://note.nkmk.me/python-list-ndarray-dataframe-normalize-standardize/
    @staticmethod
    def standarize(data,ddof=False):
        stddat = data - data.mean() / data.std(ddof=ddof)
        stddat.index = data.index
        stddat.columns = data.columns
 
        return stddat
     
    #ιθ»’οΌει¨ι’ζ°οΌ
    def __reverse(self,value):
        #if self.rtype is "likert" :
        return (self.likert_[0]+self.likert_[1]) - value
     
    #ε ε­θ² θ·ιγγι η?γγΌγΏγιΈζγγοΌει¨ι’ζ°οΌ
    def __select_items_by_loading(self,loading,data):
        # ζ Όη΄η¨θΎζΈοΌpandas.DataFrameγ«ε€ζγγγ¨γγ«γεγγΌγΏγ¨γγγγθΎζΈγ«
        item_dict = {} 
        # ε ε­θ² θ·ιγ?η΅Άε―Ύε€γιΎε€γγε€§γγθ³ͺεγεγεΊγ
        for i in range(len(loading)):
            if loading[i] >= self.tol_loading_ :
                item_dict[data.columns[i]] = data.values[:,i]
            elif loading[i] <= -self.tol_loading_ :
                item_dict[data.columns[i]] = self.__reverse(data.values[:,i])
        return pd.DataFrame(item_dict)
         
    #ε ε­θ² θ·ιγγγ―γ­γ³γγγ―Ξ±γθ¨η?γγοΌει¨ι’ζ°οΌ
    def __loading_to_alpha(self,loading,data):
        items = self.__select_items_by_loading(loading,data)
        #γ―γ­γ³γγγ―Ξ±γ?θ¨η?
        alpha = 0e0
        if len(items.columns) > 1 : # οΌθ³ͺεγ?ε ΄εγ―γε±ιε ε­γ―γͺγγ£γγ¨θ¦γͺγ
            alpha = self.cronbach_alpha(items,ddof=self.ddof_)
        #print(item_dict.keys()," alpha="+str(alpha))
        return alpha
 
    #I-TηΈι’γθ¨η?γγ
    def __loading_to_itcorr(self,loading,data):
        items = self.__select_items_by_loading(loading,data)
        itcorr= self.item_total_corr(items)
        return itcorr
     
    #ζεζ³γθ¨η?γγ
    def __loading_to_rho(self,loading,data):
        items = self.__select_items_by_loading(loading,data)
        rho   = self.split_half_rho(items)
        return rho
     
    #γ½γΌγζΈε ε­θ² θ·ι
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
     
    #ε ε­ζ°ζ¨ε?
    def explore( self, data ):
     
        #ζ¨ζΊε
        stddat = self.standarize(data,ddof=self.ddof_)
     
        #ζε€§ε ε­ζ°ζ¨ε?
        ev = pd.DataFrame(np.linalg.eigvals(stddat.corr()))
        max_factors = 0
        for e in ev[0]:
            if e > self.tol_scree_ :
                max_factors += 1
        print('predicted max_factors = '+str(max_factors))
     
        #ε ε­ζ°ζ’η΄’γ«γΌγ
        nfactors = max_factors
        while nfactors > 0 :
         
            print('try nfactors = '+str(nfactors))
             
            #ε ε­εζ
            fa = FactorAnalyzer(nfactors,rotation=self.rotation_,method=self.method_,
                                bounds=self.bounds_,impute=self.impute_)
            fa.fit(stddat) # ε ε­εζοΌηΉη°ε€εθ§£SVDοΌ
     
            #ε ε­θ² θ·ι(ε ε­Γθ³ͺεοΌ
            #loadings = pd.DataFrame(fa.components_.T)
            loadings = pd.DataFrame(fa.loadings_)
             
            #δΏ‘ι Όζ§οΌε¦₯ε½ζ§οΌεηδΈθ²«ζ§οΌγ?γγ§γγ―
            ifactor = nfactors - 1
            while ifactor >= 0:
                alpha  = self.__loading_to_alpha(loadings.iloc[:,ifactor],data)
                if alpha < self.tol_alpha_ : # ιΎε€ζͺζΊγͺγδΈθ²«ζ§γγͺγγγγε ε­ζ°γε¦₯ε½γ§γ―γͺγγ
                    break
                ifactor -= 1
 
            #ζ’η΄’η΅δΊζ‘δ»ΆοΌε¨ε ε­γ?ε¦₯ε½ζ§γη’Ίθͺγγγε ΄εγζ’η΄’γζγ‘εγ
            if ifactor < 0 : 
                print('found optimum nfactors = '+str(nfactors))
                break
         
            #ζ’η΄’ηΆηΆζ‘δ»ΆοΌε ε­γη’Ίθͺγ§γγͺγγ£γε ΄εγε ε­ζ°γζΈγγγ¦ζ¬‘γΈγ
            nfactors -= 1
         
        #ζ¨ε?ε ε­ζ°οΌγγ0γͺγε€±ζοΌ
        if nfactors > 0:
            print("Predicted nfactors = "+str(nfactors))
        else:
            print("Prediction error : No factors in this data, because nfactors = 0")
         
        #γ€γ³γΉγΏγ³γΉε€ζ°γ«ζ Όη΄
        self.max_factors_    = max_factors
        self.nfactors_       = nfactors
        self.scree_eigvals_  = ev
         
        return nfactors # ζε nfactors>0, ε€±ζ nfactors=0
     
    def scree_plot(self):
        # εΊζΊη·(εΊζε€1)
        ev_1 = np.ones(len(self.scree_eigvals_))
 
        # ε€ζ°
        plt.plot(self.scree_eigvals_, 's-')   # δΈ»ζεεζγ«γγεΊζε€
        plt.plot(ev_1, 's-') # γγγΌγγΌγΏ
 
        # θ»Έε
        plt.xlabel("ε ε­γ?ζ°")
        plt.ylabel("εΊζε€")
 
        plt.grid()
        plt.show()
 
    #ε ε­εζγ?θ§£θͺ¬
    #
    def analyze(self,data,nfactors):
 
        #γ¨γ©γΌε¦η
        if nfactors < 1 :
            raise ValueError('NFACTORS in analyze() must be more than zero!')
             
        #ζ¨ζΊε
        stddat = self.standarize(data,ddof=self.ddof_)
 
        #ε ε­εζοΌMINRESζ³γδ½Ώη¨οΌ
        fa = FactorAnalyzer(nfactors,rotation=self.rotation_,method=self.method_,
                            bounds=self.bounds_,impute=self.impute_)
        fa.fit(stddat)
 
        #δ»?ε ε­εηζ "Fn"
        factnames = [] # ε ε­ειε
        i=0
        while i < nfactors: #ε ε­γ©γγ«ηζ
            factname = 'F' + str(i+1)
            factnames.append(factname)
            i+=1
             
        #ε ε­θ² θ·ιοΌθ³ͺεΓε ε­οΌ
        self.loadings_ = pd.DataFrame(fa.loadings_,index=data.columns,columns=factnames)
        self.sorted_loadings_ = self.sort_loadings(self.loadings_,tol=self.tol_loading_)
 
        #εηδΈθ²«ζ§
        self.alphas_ = pd.DataFrame(np.zeros(nfactors).reshape(1,nfactors),index=["alpha"],columns=factnames)
        for i in range(nfactors):
            alpha = self.__loading_to_alpha(self.loadings_.iloc[:,i],data)
            self.alphas_.iloc[0,i] = alpha
         
        #I-TηΈι’
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
 
        #ε ε­εΎηΉ
        self.factors_ = pd.DataFrame(fa.transform(stddat),index=data.index,columns=factnames)  # ε ε­εΎηΉγ«ε€ζ
 
        #ε±ιζ§οΌε±ιεζ£γ?ε€γ?γγ¨γθ€ζ°γ?θ¦³ζΈ¬ε€ζ°γ«ε½±ιΏγγεΊ¦εγοΌ
        x = fa.get_communalities()
        if x is not None :
            self.communalities_ = pd.DataFrame(x,index=data.columns,columns=['comm.'])
            #print(self.communalities_)
         
        #γͺγͺγΈγγ«ηΈι’γ?εΊζε€γε ε­ηΈι’γ?εΊζε€
        x = fa.get_eigenvalues()
        if x is not None :
            self.eigenvalues_ = pd.DataFrame(x,index = ['original','factor'],columns = data.columns)
            #print(self.eigenvalues_)
         
        #ε ε­ε―δΈγε ε­ε―δΈηγη΄―η©ε ε­ε―δΈη
        x = fa.get_factor_variance()
        if x is not None :
            self.factor_variance_ = pd.DataFrame(x,index=['var.','prop.','cumu.'],columns = factnames)
            #print(self.factor_variance_)
 
        #η¬θͺζ§οΌη¬θͺεζ£γ?ε€γ?γγ¨γεζ°γ?θ¦³ζΈ¬ε€ζ°γ«ε½±ιΏγγεΊ¦εγοΌ
        x = fa.get_uniquenesses()
        if x is not None :
            self.uniquenesses_ = pd.DataFrame(x,index = data.columns,columns = ['uniq.'])
            #print(self.uniquenesses_)
 
        #ηΈι’θ‘εοΌι η?ιγ?ηΈι’οΌ
        self.corr_ = pd.DataFrame(fa.corr_,index=data.columns,columns=data.columns)
        #print(self.corr_)
         
        #εθ»’θ‘εοΌζδΊ€εθ»’γ?ε ΄εγ γθ¨η?γγγθ»Έγ?εθ»’γθ‘¨γοΌ
        if hasattr(fa,'rotation_matrix_') :
            self.rotation_matrix_ = pd.DataFrame(fa.rotation_matrix_,index=factnames,columns=factnames)
            #print(self.rotation_matrix_)
        else:
            print("'fa' does not have a menber 'rotation_matrix_'")
         
        #ζ§ι θ‘εοΌpromaxγ?ε ΄εγ γθ¨η?γγγοΌ
        if hasattr(fa,'structure_'):
            self.structure_ = pd.DataFrame(fa.structure_,index=data.columns,columns=factnames)
            #print(self.structure_)
        else:
            print("'fa' does not have a menber 'structure_'")
 
         
        #ε ε­ηΈι’θ‘εοΌζδΊ€εθ»’γ?ε ΄εγ γθ¨η?γγγγpromaxγ§θ¨η?γγγͺγγ?γ―γγ°οΌοΌ
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
