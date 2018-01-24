import numpy as np 

def partitionrnd(n, prop, g):
    """Generate a random partition.
    n    : number of instances 
    prop : proportion of clusters
    g    : number of clusters
    """
    tmp = np.random.uniform(low=0.0, high=1.0, size=n)
    z_i = g + 1 - np.sum(np.tile(tmp,(g,1)).T < np.tile(np.cumsum(prop),(n,1)), axis = 1)
    return z_i

def generatedata(mu, n=100, d=100, g=2, m=2, prop_r = None, prop_c = None, noise=1):
	"""Generate a data matrix with a gxm bloc structure.
    n      : number of rows
	d      : number of columns
    prop_r : proportion of row clusters
    prop_c : proportion of column clusters
    g      : number of row clusters
    m      : number of column clusters
	mu     : a gxm matrix with the mean of each block
    Return the matrix and the partitions
    """
	g, m = mu.shape

	if prop_r is None:
		prop_r = (1/g * np.ones(g))
	if prop_c is None:
		prop_c = (1/m * np.ones(m))

	s = (g,m)
	sigma = 0.1*np.ones(s)*noise

	# Generate one partition for instances and one for variables
	z_i = partitionrnd(n, prop_r, g)
	w_j = partitionrnd(d, prop_c, m)

	# Generate the data matrix x 
	s = (n, d)
	data = np.zeros(s)

	for i in range(n):
		for j in range(d):
			data[i][j] = np.random.normal(mu.item(z_i[i]-1,w_j[j]-1), sigma.item(z_i[i]-1,w_j[j]-1),1)[0]
	
	return data, z_i, w_j