import numpy  as np
import pandas as pd
import lingam
import semopy
import graphviz
#import causal_discover as cd

####################################################################
###
### Scaing Functions for pandas.DataFrame
###
###   1. Use standardize() for regular distribution data such as Likert N-item questionnaire. 
###   2. Use normalize() for the data bounded between upper limit and lower one.
###   3. Use scale_by() for dynamics data such as time-plot data.
###
####################################################################
def standardize(data,ddof=False):
	stddat = ( data - data.mean() ) / data.std(ddof=ddof)
	stddat.index = data.index
	stddat.columns = data.columns
	return stddat

def normalize(data):
	stddat = ( data - data.min() ) / ( data.max() - data.min() )
	stddat.index = data.index
	stddat.columns = data.columns
	return stddat

def scale_by(data,row=0):
	stddat = data / data.iloc[row]
	stddat.index = data.index
	stddat.columns = data.columns
	return stddat

####################################################################
###
### Causal Finder using SEM
###
###     by comparing Fitness on Single Path Regression
###
####################################################################
class SEMCausalFinder:

	def __init__(self,alpha=0.05,eps_cfi=0.05,eps_gfi=0.05,eps_nfi=0.05):
		self.adjacency_matrix_ = None
		self.p_values_matrix_  = None
		self.stdests_          = None
		self.pvalues_          = None
		self.cfis_             = None
		self.gfis_             = None
		self.nfis_             = None
		self.aics_             = None
		self.rmseas_           = None
		self.adjmax_           = None
		self.alpha_            = alpha   # significant level for p-value (default:0.05)
		self.eps_cfi_          = eps_cfi # significant difference level for CFI (default:0.05)
		self.eps_gfi_          = eps_gfi # significant difference level for GFI (default:0.05)
		self.eps_nfi_          = eps_nfi # significant difference level for NFI (default:0.05)

	######################
	### STATIC METHODS ###
	######################

	@staticmethod
	def print_full(df,title="No title"):
		x = ""
		if df is not None:
			x  = f"{title}:\n"
			pd.set_option('display.max_columns',len(df.columns))
			pd.set_option('display.max_rows',len(df.index))
			x += str(df)
			x += "\n\n"
		return x

	########################
	### INTERNAL METHODS ###
	########################

	def __make_sem_combination_matrix(self,data):
		n = len(data.columns)
		alpha   = self.alpha_ 
		eps_cfi = self.eps_cfi_
		eps_gfi = self.eps_gfi_
		eps_nfi = self.eps_nfi_
		varnames      = data.columns.to_list()
		stdest_matrix = np.zeros((n,n))
		pvalue_matrix = np.ones((n,n))
		cfi_matrix    = np.zeros((n,n))
		gfi_matrix    = np.zeros((n,n))
		nfi_matrix    = np.zeros((n,n))
		aic_matrix    = np.zeros((n,n))
		rmsea_matrix  = np.ones((n,n))
		adj_matrix    = np.zeros((n,n))
		for i in range(n):
			lhs = varnames[i];
			for j in range(i+1,n,1):
				rhs = varnames[j];
				#print(f"{lhs} ~ {rhs}")
				model_l = semopy.Model(f"{lhs} ~ {rhs}\n") # lhs <- rhs
				model_r = semopy.Model(f"{rhs} ~ {lhs}\n") # lhs -> rhs
				model_l.fit(data)
				model_r.fit(data)
				try:
					inspect_l = model_l.inspect(std_est=True)
					stdest_l = inspect_l["Est. Std"][0]
					pvalue_l = inspect_l["p-value"][0]
				except np.linalg.LinAlgError as e:
					print(f"NumPy Linear Algebra Error : {lhs} ~ {rhs}")
					stdest_l = 0.0
					pvalue_l = 1.0
				try:
					inspect_r = model_r.inspect(std_est=True)
					stdest_r = inspect_r["Est. Std"][0]
					pvalue_r = inspect_r["p-value"][0]
				except np.linalg.LinAlgError as e:
					print(f"NumPy Linear Algebra Error : {rhs} ~ {lhs}")
					stdest_r = 0.0
					pvalue_r = 1.0
				goodfit_l = semopy.calc_stats(model_l)
				goodfit_r = semopy.calc_stats(model_r)
				#
				cfi_l    = goodfit_l["CFI"][0]
				cfi_r    = goodfit_r["CFI"][0]
				gfi_l    = goodfit_l["GFI"][0]
				gfi_r    = goodfit_r["GFI"][0]
				nfi_l    = goodfit_l["NFI"][0]
				nfi_r    = goodfit_r["NFI"][0]
				aic_l    = goodfit_l["AIC"][0]
				aic_r    = goodfit_r["AIC"][0]
				rmsea_l  = goodfit_l["RMSEA"][0]
				rmsea_r  = goodfit_r["RMSEA"][0]
				if   pvalue_l < alpha and pvalue_r < alpha : # 両向き有意な場合
				
					if abs(cfi_l-cfi_r) > eps_cfi or abs(gfi_l-gfi_r) > eps_gfi or abs(nfi_l-nfi_r) > eps_nfi : # 左右差がある場合
						if cfi_l > cfi_r : # 左向き確定
							adj_matrix[i,j] = stdest_l
						else : # 右向き
							adj_matrix[j,i] = stdest_r
					else: # 左右差がない場合 --> 両向き
						adj_matrix[i,j] = stdest_l
						adj_matrix[j,i] = stdest_r
				
				elif pvalue_l < alpha and pvalue_r >=alpha : # 左向きのみ有意な場合
				
					adj_matrix[i,j] = stdest_l
				
				elif pvalue_l >=alpha and pvalue_r < alpha : # 右向きのみ有意な場合
				
					adj_matrix[j,i] = stdest_r
				
				#else : # 左右どちらも有意ではない場合
				#
				#	# do nothing
				#
				#print(f"i={i},j={j},n={n}")
				stdest_matrix[i,j] = stdest_l
				stdest_matrix[j,i] = stdest_r
				pvalue_matrix[i,j] = pvalue_l
				pvalue_matrix[j,i] = pvalue_r
				cfi_matrix[i,j]    = cfi_l   
				cfi_matrix[j,i]    = cfi_r   
				gfi_matrix[i,j]    = gfi_l   
				gfi_matrix[j,i]    = gfi_r   
				nfi_matrix[i,j]    = nfi_l   
				nfi_matrix[j,i]    = nfi_r   
				aic_matrix[i,j]    = aic_l   
				aic_matrix[j,i]    = aic_r   
				rmsea_matrix[i,j]  = rmsea_l 
				rmsea_matrix[j,i]  = rmsea_r 
	
		self.stdests_ = pd.DataFrame(stdest_matrix,columns=varnames,index=varnames)
		self.pvalues_ = pd.DataFrame(pvalue_matrix,columns=varnames,index=varnames)
		self.cfis_    = pd.DataFrame(cfi_matrix   ,columns=varnames,index=varnames)
		self.gfis_    = pd.DataFrame(gfi_matrix   ,columns=varnames,index=varnames)
		self.nfis_    = pd.DataFrame(nfi_matrix   ,columns=varnames,index=varnames)
		self.aics_    = pd.DataFrame(aic_matrix   ,columns=varnames,index=varnames)
		self.rmseas_  = pd.DataFrame(rmsea_matrix ,columns=varnames,index=varnames)
		self.adjmat_  = pd.DataFrame(adj_matrix   ,columns=varnames,index=varnames)

		self.p_values_matrix_  = pvalue_matrix
		self.adjacency_matrix_ = adj_matrix

	#####################
	### CLASS METHODS ###
	#####################

	def fit(self,data):
		self.__make_sem_combination_matrix(data)
		print(self)
		return self

	def get_error_independence_p_values(self,data):
		return self.p_values_matrix_

	def __str__(self):
		x  = "\n"
		x += "Cutoff for p-value   : " + str(self.alpha_) + "\n"
		x += "Cutoff for CFI diff. : " + str(self.eps_cfi_) + "\n"
		x += "Cutoff for GFI diff. : " + str(self.eps_gfi_) + "\n"
		x += "Cutoff for NFI diff. : " + str(self.eps_nfi_) + "\n"
		x += "\n"
		x += self.print_full(self.stdests_,"Est. Stds")
		x += self.print_full(self.pvalues_,"p-values" )
		x += self.print_full(self.cfis_   ,"CFIs")
		x += self.print_full(self.gfis_   ,"GFIs")
		x += self.print_full(self.nfis_   ,"NFIs")
		x += self.print_full(self.aics_   ,"AICs")
		x += self.print_full(self.rmseas_ ,"RMSEAs")
		x += self.print_full(self.adjmat_ ,"Adjercency Matrix")
		return x


####################################################################
###
### Causal Analizer using LiNGAM and Semopy
###
###     by connecting Causal Discovery (LiNGAM) & Causal Guess (Semopy)
###
####################################################################
class CausalAnalysis:

	def __init__(self,style="lavaan",method="direct",tol=0.01,alpha=0.05,enable_cov=False):
		self.model_     = None  # SEM-model text
		self.sem_       = None  # SEM-model object
		self.lingam_    = None  # LiNGAM-model object
		self.tol_       = tol   # Threshold for Causality effectiveness (defualt=0.01)
		self.alpha_     = alpha # Significant value level for p-value (defualt=0.05)
		self.style_     = style # SEM-model format style ["lavaan"|"sem"] (default=lavaan)
		self.method_    = method# Discovery medhod ["direct"|"ica"|"var"|"varma"|"rcd"|"sem"] (default=direct)
		self.enable_cov_= enable_cov # Enable Covariance Path <--> (default=False)

	######################
	### STATIC METHODS ###
	######################

	@staticmethod
	def read_model(file,encoding='UTF-8'):
		file = open(file,'r',encoding=encoding)
		text = file.read()
		file.close()
		return text

	@staticmethod
	def write_model(file,data,encoding='UTF-8'):
		file = open(file,'w',encoding=encoding)
		file.write(data)
		file.close()

	@staticmethod
	def lingam_to_sem(data,lingam_model,tol=0.01,alpha=0.05,sem_style="lavaan",enable_cov=False):
		# 変数の数
		n        = len(data.columns)

		# 変数名リスト
		varnames = data.columns.to_list()

		# 因果係数の行列
		adjmat   = lingam_model.adjacency_matrix_

		# 変数の順序
		if hasattr(lingam_model,"causal_order_"):
			order = lingam_model.causal_order_ # 因果順序
		else:
			order = list(range(n)) # 0, 1, 2, ..., n

		# p値の行列
		if hasattr(lingam_model,"get_error_independence_p_values"):
			pvalmat = lingam_model.get_error_independence_p_values(data)
		else:
			pvalmat = np.zeros((n,n)) # ゼロ行列

		# モデルの書き方
		if sem_style == "lavaan":
			left = "~"
			right= "=~"
			both = "~~"
		elif sem_style == "sem":
			left = "<-"
			right= "->"
			both = "<>"
		else:
			print(f"lingam_to_sem: irregal option sem_style={sem_style}. ['lavaan','sem']")

		sem_model=""
		sem_model2=""
		for i in range(n):
			for j in range(i+1,n,1):
				lhs = varnames[order[i]]
				rhs = varnames[order[j]]
				if adjmat[i,j] > tol and pvalmat[i,j] < alpha :
					if adjmat[j,i] > tol and pvalmat[j,i] < alpha :
						sem_model2 += f"{lhs} {both} {rhs}\n"
					else:
						sem_model += f"{lhs} {left} {rhs}\n"
				else:
					if adjmat[j,i] > tol and pvalmat[j,i] < alpha :
						sem_model += f"{rhs} {left} {lhs}\n"
		#print(sem_model)
		if enable_cov : sem_model += sem_model2

		return sem_model

	########################
	### INTERNAL METHODS ###
	########################

	def __lingam_to_sem(self,data,lingam_model):
		return self.lingam_to_sem(data,lingam_model,tol=self.tol_,alpha=self.alpha_,sem_style=self.style_,enable_cov=self.enable_cov_)


	#####################
	### CLASS METHODS ###
	#####################

	# 因果分析を行う
	def analyze(self,data):
		model  = self.discover(data)
		print(model)
		#print(data.columns.to_list())
		result = self.evaluate(data,model)
		return result

	# 因果探索を行う 
	def discover(self,data):
		if self.method_ == "direct":
			self.lingam_ = lingam.DirectLiNGAM() 
			self.lingam_.fit(data)
		elif self.method_ == "ica":
			self.lingam_ = lingam.ICALiNGAM() 
			self.lingam_.fit(data)
		elif self.method_ == "var":
			self.lingam_ = lingam.VARLiNGAM() 
			self.lingam_.fit(data)
		elif self.method_ == "varma":
			self.lingam_ = lingam.VARMALiNGAM() 
			self.lingam_.fit(data)
		elif self.method_ == "rcd":
			self.lingam_ = lingam.RCD() 
			self.lingam_.fit(data)
		elif self.method_ == "sem":
			self.lingam_ = SEMCausalFinder() 
			self.lingam_.fit(data)
		#elif self.method_ == "lina":
		#	self.lingam_ = lingam.LiNA() 
		#	self.lingam_.fit(data)
		#elif self.method_ == "lim":
		#	self.lingam_ = lingam.LiM() 
		#	dis_con = [0] * n
		#	self.lingam_.fit(data,dis_con)
		#elif self.method_ == "resit":
		#	self.lingam_ = lingam.RESIT() 
		#	self.lingam_.fit(data)
		else :
			print(f"discover : Invalid option self.method_ = {self.method_}")
		self.model_ = self.__lingam_to_sem(data,self.lingam_)
		return self.model_

	# 因果推論を行う
	def evaluate(self,data,model=None):
		if model is None: model = self.model_
		if model is None: raise ValueError("No model")  
		if str(model) == "": raise ValueError("Empty model")
		self.sem_ = semopy.Model(model)
		res = self.sem_.fit(data)
		return res

	# モデルや図をファイルに書き込む
	def save(self,init_imgfile="found_causality.png",final_imgfile="evaled_causality.png",
                      init_dotfile="found_causality.gv", final_dotfile="evaled_causality.gv",
                      modfile="sem_model.txt",
                      basedir=".",show=False):
		if self.lingam_ is not None:
			dotdata = lingam.utils.make_dot(self.lingam_.adjacency_matrix_)
			graph   = graphviz.Source(dotdata,basedir+"/"+init_dotfile,format='png')
			#graph.render(outfile=basedir+"/"+init_imgfile,view=show)
			graph.render(view=show)
		if self.model_ is not None:
			self.write_model(basedir+"/"+modfile,self.model_)
		if self.sem_ is not None:
			self.write_model(basedir+"/"+final_dotfile,str(dotdata))
			try:
				dotdata = semopy.semplot(self.sem_,basedir+"/"+final_imgfile,plot_covs=True,std_ests=True,show=show)
			except np.linalg.LinAlgError as e:
				print(f"NumPy Linear Algebra Error : Graph can not be drawn.")

	# インスタンスをダンプする
	def __str__(self):
		x  = "\n"
		x += "SEM Model Style   : " + str(self.style_) + "\n"
		x += "LiNGAM Method     : " + str(self.method_) + "\n"
		x += "Cutoff Causality  : " + str(self.tol_) + "\n"
		x += "Cutoff p-value    : " + str(self.alpha_) + "\n"
		x += "\n"
		if self.lingam_ is not None:
			if hasattr(self.lingam_,"causal_order_"):
				x += "Causal Order:\n"
				x += str(self.lingam_.causal_order_)
				x += "\n\n"
			x += "Causal Matrix:\n"
			x += str(self.lingam_.adjacency_matrix_)
			x += "\n\n"
		if self.model_ is not None:
			x += "SEM Model:\n"
			x += str(self.model_)
			x += "\n\n"
		if self.sem_ is not None:
			try:
				ins = self.sem_.inspect(std_est=True)
				pd.set_option('display.max_columns',len(ins.columns))
				pd.set_option('display.max_rows',len(ins.index))
				x += "Infomation:\n"
				x += str(ins)
				x += "\n\n"
			except np.linalg.LinAlgError as e:
				print(f"NumPy Linear Algebra Error : Infomationcan not be shown.")
			x += "Fitness Index:\n"
			gfi = semopy.calc_stats(self.sem_)
			pd.set_option('display.max_columns',len(gfi.index))
			pd.set_option('display.max_rows',len(gfi.columns))
			x += str(gfi.T)
			x += "\n\n"

		return x

