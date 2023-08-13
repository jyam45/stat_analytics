from scipy.stats import norm, mvn
from scipy.stats import multivariate_normal
from scipy.optimize import minimize, minimize_scalar
from scipy.optimize import Bounds, BFGS
import numpy as np
import semopy.polycorr as so

# カテゴリー変数を出現回数行列に変換する
def frequency_table(x,y):
	"""
	Create a frequency table by cross-tabulatin of x by y.

	Parameters
	----------
	x : pd.DataFrame
		An array of categorial data
	y : pd.DataFrame
		An array of categorial data
	
	Returns
	-------
	n : np.ndarray
		A frequency matrix
	"""
	r = max(x) - min(x) + 1 
	s = max(y) - min(y) + 1
	n = np.zeros((r,s))
	i0 = min(x)
	j0 = min(y)
	for i, j in zip(x,y):
		n[ i-i0, j-j0 ] += 1
	#print(n)
	return n

# 正規分布の境界値を生成する
def estimate_boundaries( x, inf=10 ):
	"""
	Compute boundary thresholds by inversion CDF.

	Parameters
	----------
	x : pd.DataFrame
		An array of categorial data
	inf : int, optional
		inifinity value on normal distribution. Default value is 10.
	
	Returns
	-------
	a : np.ndarray
		An array of boundary thresholds.
	cat : np.ndarray
		An array of category numbers.
	"""
	cat, cnt = np.unique(x,return_counts=True)
	n = cnt.sum()
	a = norm.ppf( cnt.cumsum() / n )[:-1]
	a = np.append( -inf, np.append( a, inf ) )
	#print(a)
	return a, cat

# ２次元ガウス分布（二変量正規分布）から組み合わせが得られる確率を計算する
def probability_table(a,b,rho,means=[0,0],var=[1,1],eps=1e-15,deriv=False):
	"""
	Compute the probabilities than observations fall into cells.

	Parameters
	----------
	a : np.ndarray
		An array of cell boundary thresholds of data x
	b : np.ndarray
		An array of cell boundary thresholds of data y
	rho : float
		A correlation value as off-diagonal elements of 2x2 covariant matrix
	means : list, float, optional
		2 mean values, default values are [0,0] (standardized).
	var : list, float, optional
		Diagonal elements of 2x2 covariant matrix, default values are [1,1].
	
	Returns
	-------
	pi : np.ndarray
		A matrix of probabilities in cell.
	"""
	cov = np.array([[var[0],rho],[rho,var[1]]])
	m = len(a) - 1
	n = len(b) - 1
	pi = np.zeros((m,n))
	phi2 = multivariate_normal(cov=cov)
	if deriv :
		dpi = np.zeros((m,n))
		for i in range(m):
			for j in range(n):
				#a1b1 = phi2.cdf( [a[i+1],b[j+1]] )
				#a1b0 = phi2.cdf( [a[i+1],b[j  ]] )
				#a0b1 = phi2.cdf( [a[i  ],b[j+1]] )
				#a0b0 = phi2.cdf( [a[i  ],b[j  ]] )
				#pival = a1b1 - a1b0 - a0b1 + a0b0
				pival = mvn.mvnun([a[i],b[j]],[a[i+1],b[j+1]],means,cov)[0]
				pi[i,j] = max( pival, eps )
				da1b1 = phi2.pdf( [a[i+1],b[j+1]] )
				da1b0 = phi2.pdf( [a[i+1],b[j  ]] )
				da0b1 = phi2.pdf( [a[i  ],b[j+1]] )
				da0b0 = phi2.pdf( [a[i  ],b[j  ]] )
				dpi[i,j] = da1b1 - da1b0 - da0b1 + da0b0
		return pi,dpi
	else:
		for i in range(m):
			for j in range(n):
				#a1b1 = phi2.cdf( [a[i+1],b[j+1]] )
				#a1b0 = phi2.cdf( [a[i+1],b[j  ]] )
				#a0b1 = phi2.cdf( [a[i  ],b[j+1]] )
				#a0b0 = phi2.cdf( [a[i  ],b[j  ]] )
				#pival = a1b1 - a1b0 - a0b1 + a0b0
				pival = mvn.mvnun([a[i],b[j]],[a[i+1],b[j+1]],means,cov)[0]
				pi[i,j] = max( pival, eps )
		return pi	

def check_deriv(a,b,n):
	def calc_f(rho):
		pi = probability_table(a,b,rho)
		f   = -( n * np.log(pi) ).sum()
		return f
	def calc_fg(rho):
		pi,dpi  = probability_table(a,b,rho,deriv=True)
		f   = -( n * np.log(pi) ).sum()
		dfdr= -( n *dpi / pi ).sum()
		return f,dfdr
	ub=1.0
	lb=-1.0
	maxpoints=50
	print("DERIV CHECK")	
	for i in range(maxpoints):
		rho = lb + (i+1)/maxpoints *(ub-lb)
		f0,g0 = calc_fg(rho)
		delta = 1e-1
		while delta > 1e-14:
			f1 = calc_f(rho+delta)
			g1 = (f1-f0)/delta
			if abs(g1-g0) < 1e-4 * abs(g0):
				break
			delta *= 0.1
		print(i," rho=",rho," delta=",delta," g_exact=",g0," g_numer=",g1," delta_g=",abs(g1-g0)/abs(1e-14+g0))

		

# 尤度関数の微分
def der_likelihood(a,b,rho,n,pi,dpi,loc=0,scale=1):
	"""
	Compute derivatives of likelihood function.

	Parameters
	----------
	a : np.ndarray
		An array of cell boundary thresholds of data x
	b : np.ndarray
		An array of cell boundary thresholds of data y
	rho : float
		A correlation value as off-diagonal elements of 2x2 covariant matrix
	n : np.ndarray
		Two dimensional frequency table.
	pi : np.ndarray
		Two dimensional probability table.
	dpi : np.ndarray
		Two dimensional probability derivatives table.
	loc : float, optional
		A center value of cumulative distribution function (CDF) [default=0]
	scale : float, optional
		A coefficient value of cumulative distribution function (CDF) [default=1]
	
	Returns
	-------
	dfdr : float
		A derivative for likelihood function by rho.
	dfda : np.ndarray
		Derivatives for likelihood function by a.
	dfdb : np.ndarray
		Derivatives for likelihood function by b.
	"""
	s = len(a)
	r = len(b)
	sigma = np.sqrt(1-rho*rho)
	npi = n / pi # size=(s-1,r-1)
	npidpi = npi * dpi
	# dfdr
	dfdr = -( npi * dpi ).sum() 
	# dfda
	dfda = np.zeros((s))
	phi_a = norm.pdf(a)
	for k in range(1,s-1):
		for j in range(1,r):
			cdf00 = norm.cdf( (b[j  ]-rho*a[k])/sigma )
			cdf10 = norm.cdf( (b[j-1]-rho*a[k])/sigma )
			dfda[k] += ( npi[k-1,j-1] - npi[k,j-1] ) * phi_a[k] *  ( cdf00 - cdf10 )
	# dfdb
	dfdb = np.zeros((r))
	phi_b = norm.pdf(b)
	for m in range(1,r-1):
		for i in range(1,s):
			cdf00 = norm.cdf( (a[i  ]-rho*b[m])/sigma )
			cdf10 = norm.cdf( (a[i-1]-rho*b[m])/sigma )
			dfdb[m] += ( npi[i-1,m-1] - npi[i-1,m] ) * phi_b[m] *  ( cdf00 - cdf10 )
	return dfdr,dfda,dfdb

# 単一の相関値を計算する
def corr_single(x,y,method=None,grad=None,options=None,inf=10,maxval=0.9999,verbose=False):
	"""
	Compute a poychoric correlation value between categorical data x and y.

	Parameters
	----------
	x : np.ndarray
		An array of categorical data x
	y : np.ndarray
		An array of categorical data y
	method : str or None, optional
		Type of solver for scipy.optimize.minimize. 
		Type should be one of
		* 'Nelder-Mead'
		* 'Powell'
		* 'CG'
		* 'BFGS'
		* 'Newton-CG'
		* 'L-BFGS-B'
		* 'TNC'
		* 'COBYLA'
		* 'SLSQP'
		* 'trust-constr'
		* 'dogleg'
		* 'trust-ncg'
		* 'trust-exact'
		* 'trust-krylov'
		* None - call approimated algorithm
		see also, SciPy optimize API reference. 
	grad : str of None, optional
		Type of gradient calculation method for the solvers using gradients.
		* 'numeric' - use numerical gradients g={f(x+dx)-f(x)}/dx
		* None - use exact derivatives of likelihood function.
		Default is None
	options: dict, optional
		Options for solvers including solver specific options.	
	inf : int, optional
		inifinity value on normal distribution. Default value is 10.
	maxval: float, optional
		upper/lower bound values of a correlation rho.
		Default is 0.9999.
	verbose: bool, optional
		If True, print more optimization messages.
	
	Returns
	-------
	rho : float
		A correlation value.
	"""
	# vecter packager
	def pack(_rho,_a,_b):
		_s = len(_a)
		_r = len(_b)
		_z = np.zeros((1+_s-2+_r-2))
		_z[0]            = _rho
		_z[1:_s-1]       = _a[1:_s-1]
		_z[_s-1:_s+_r-3] = _b[1:_r-1]
		return _z
	# vecter unpackager
	def unpack(_z,_s,_r):
		if len(_z) != _s+_r-3:
			raise(ValueError("Invalid Size : len(_z) == _s + _r - 3"))
		_rho = _z[0]
		_a   = _z[1:_s-1]
		_b   = _z[_s-1:_s+_r-3]
		_a   = np.append( -inf, np.append( _a, inf ) )
		_b   = np.append( -inf, np.append( _b, inf ) )
		return _rho,_a,_b
	def make_bounds(_s,_r):
		_n = _s-2 + _r-2
		lb = np.zeros((1+_n))
		ub = np.zeros((1+_n))
		kp = np.empty((1+_n),dtype=bool)
		lb[0]      = -1.0
		lb[1:_n+1] = -inf	
		ub[0]      =  1.0
		ub[1:_n+1] =  inf	
		kp[0]      =  True
		kp[1:_n+1] =  False
		bounds = Bounds(lb,ub,keep_feasible=kp)
		return bounds

	a, a_cat = estimate_boundaries(x,inf=inf)
	b, b_cat = estimate_boundaries(y,inf=inf)
	n = frequency_table(x,y)

	if method is None :

		def fun_likelihood(rho):
			pi = probability_table(a,b,rho)
			f  = -( n * np.log(pi) ).sum()
			#print("rho=",rho," f=",f)
			return f
		res = minimize_scalar( fun_likelihood,bounds=(-1,1),args=(),method="bounded")
		if res.success :
			return res.x
		else:
			print(res)
			raise(ValueError("Optimization is falure."))

	elif method == "deriv_check":

		check_deriv(a,b,n)
		raise(ValueError("Only do derivertive checker."))

	else:
		s = len(a)
		r = len(b)
		z_init = pack( 0.0, a, b )

		def f_mlikelihood(z):
			#print("z=\n",z)
			#print("a*dz=\n",z-z_init)
			rho,aa,bb = unpack(z,s,r)
			rho = max(min(rho,maxval),-maxval)
			pi  = probability_table(aa,bb,rho)
			f   = -( n * np.log(pi) ).sum()
			#print("rho=",rho," f=",f)
			return f
		def df_mlikelihood(z):
			#print("z=\n",z)
			rho,aa,bb = unpack(z,s,r)
			rho = max(min(rho,maxval),-maxval)
			pi,dpi = probability_table(aa,bb,rho,deriv=True)
			dfdr,dfda,dfdb  = der_likelihood(aa,bb,rho,n,pi,dpi)
			df = pack(dfdr,dfda,dfdb)
			#print("dfdr=\n",dfdr)
			#print("dfda=\n",dfda)
			#print("dfdb=\n",dfdb)
			return df

		func = f_mlikelihood

		if method == "CG" or \
		   method == "BFGS" or \
		   method == "Newton-CG" or \
		   method == "L-BFGS-B" or  \
		   method == "TNC" or  \
		   method == "SLSQP" or  \
		   method == "dogleg" or  \
		   method == "trust-ncg" or \
		   method == "trust-krylov" or \
		   method == "trust-exact" or \
		   method == "trust-constr" :
			if grad is None:
				jacobian = df_mlikelihood
			elif grad == "numeric":
				jacobian = "2-piont"
		else:
			jacobian = None

		if method == "Nelder-Mead" or \
		   method == "L-BFGS-B" or \
		   method == "TNC" or \
		   method == "SLSQP" or \
		   method == "Powell" or \
		   method == "trust-constr" or \
		   method == "COBYLA":
			bounds = make_bounds(s,r)
		else:
			bounds = None

		if options is None:
			if verbose :
				opts = {'disp':True,'return_all':True}
			else:
				opts = {}# {'disp':True,'return_all':True}
			if method == "Powell" or \
			   method == "SLSQP"  :
				opts["ftol"] = 1e-8
			if method == "Newton-CG" :
				opts["xtol"] = 1e-8
			if method == "BFGS" or \
			   method == "CG" or \
			   method == "L-BFGS-B":
				opts["gtol"] = 1e-6
		else:
			opts = options

		res = minimize( func, z_init, jac=jacobian, method=method, bounds=bounds, options=opts)

		if res.success :
			#print(res)
			return res.x[0]
		else:
			print(res)
			raise(ValueError("Optimization is falure."))

		# Converged pattern
		#res = minimize( f_mlikelihood,z_init,jac="2-point",method="SLSQP",bounds=bounds,
		#                options={'ftol':1e-8,'eps':1e-6,'disp':True})
		#res = minimize( f_mlikelihood,z_init,method="Powell",
		#                options={'ftol':1e-8,'xtol':1e-12,'disp':True,'return_all':True})
		#res = minimize( f_mlikelihood,z_init,jac="2-point",method="BFGS",
		#                options={'gtol':1e-6,'xrtol':1e-8,'disp':True,'return_all':True} )
		#res = minimize( f_mlikelihood,z_init,jac=df_mlikelihood,method="SLSQP",bounds=bounds,
		#                options={'ftol':1e-8,'disp':True})
		#res = minimize( f_mlikelihood,z_init,jac=df_mlikelihood,method="L-BFGS-B",bounds=bounds,
		#                options={'gtol':1e-6,'disp':True})
		#res = minimize( f_mlikelihood,z_init,jac=df_mlikelihood,method="L-BFGS-B",
		#                options={'gtol':1e-6,'disp':True})

		# Not converged pattern
		#res = minimize( f_mlikelihood,z_init,jac=df_mlikelihood,method="BFGS",
		#                options={'gtol':1e-6,'xrtol':1e-4,'disp':True,'return_all':True})


# 相関行列を計算する
def corr(data,method=None,grad=None,options=None,inf=10,maxval=0.9999,verbose=False):
	"""
	Compute a poychoric correlation matrix of categorical data.

	Parameters
	----------
	data : pd.DataFrame
		An array of categorical data x
	method : str or None, optional
		Type of solver for scipy.optimize.minimize. 
		Type should be one of
		* 'Nelder-Mead'
		* 'Powell'
		* 'CG'
		* 'BFGS'
		* 'Newton-CG'
		* 'L-BFGS-B'
		* 'TNC'
		* 'COBYLA'
		* 'SLSQP'
		* 'trust-constr'
		* 'dogleg'
		* 'trust-ncg'
		* 'trust-exact'
		* 'trust-krylov'
		* 'semopy' - use semopy's polychoric_corr() function
		* None - use an approimated algorithm
		see also, SciPy optimize API reference. 
	grad : str of None, optional
		Type of gradient calculation method for the solvers using gradients.
		* 'numeric' - use numerical gradients g={f(x+dx)-f(x)}/dx
		* None - use exact derivatives of likelihood function.
		Default is None
	options: dict, optional
		Options for solvers including solver specific options.	
	inf : int, optional
		inifinity value on normal distribution. Default value is 10.
	maxval: float, optional
		upper/lower bound values of a correlation rho.
		Default is 0.9999.
	verbose: bool, optional
		If True, print more optimization messages.
	
	Returns
	-------
	corr : np.ndarray
		A correlation matrix.
	"""

	n = len(data.columns)
	corr = np.eye(n)
	if method == "semopy" :
		for i in range(0,n):
			x = data.iloc[:,i]
			for j in range(i+1,n):
				y  = data.iloc[:,j]
				rho = so.polychoric_corr(x,y)
				corr[i,j] = corr[j,i] = rho
	else:
		for i in range(0,n):
			x = data.iloc[:,i]
			for j in range(i+1,n):
				y  = data.iloc[:,j]
				rho = corr_single(x,y,method=method,grad=grad,options=options,
				                  inf=inf,maxval=maxval,verbose=verbose)
				corr[i,j] = corr[j,i] = rho
	return corr



