
# coding: utf-8

# In[50]:

import numpy as np
import healpy
import os
import pylab
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pickle
from scipy import stats
import scipy.linalg as la
import pymaster as nmt
import pyccl as ccl


# In[51]:

nsim = int(len(os.listdir('/Users/Joseph/Brookhaven/Research/sim_data/'))/2)

b = nmt.NmtBin(2048,nlb=50)
ell_arr=b.get_effective_ells()

max_ell = 1500
min_ell_bin = 1
max_ell_bin = (np.abs(ell_arr-max_ell)).argmin()

data_qso = pickle.load(open('/Users/Joseph/Brookhaven/Research/pickled/cps_qso.pkl','rb'))[0][min_ell_bin:max_ell_bin]
data_dla = pickle.load(open('/Users/Joseph/Brookhaven/Research/pickled/cps_dla.pkl','rb'))[0][min_ell_bin:max_ell_bin]

ell_arr=ell_arr[min_ell_bin:max_ell_bin]

sim_qso = np.zeros((max_ell_bin-min_ell_bin,nsim))
sim_dla = np.zeros((max_ell_bin-min_ell_bin,nsim))
for i in range(nsim):
    sim_qso[:,i] = pickle.load(open('/Users/Joseph/Brookhaven/Research/sim_data/cps_qso_'+str(i).zfill(2)+'.pkl','rb'))[0][min_ell_bin:max_ell_bin]
    sim_dla[:,i] = pickle.load(open('/Users/Joseph/Brookhaven/Research/sim_data/cps_dla_'+str(i).zfill(2)+'.pkl','rb'))[0][min_ell_bin:max_ell_bin]

error_qso = np.std(sim_qso,axis=1)
error_dla = np.std(sim_dla,axis=1)

dla = open('/Users/Joseph/Brookhaven/Research/DLA_DR12_v1.dat')
dla_ra = []; dla_dec = []; dla_z = []; qso_z = []

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

for line in dla:
    ra = [line.split()][0][2]
    dec = [line.split()][0][3]
    d = [line.split()][0][9]
    q = [line.split()][0][4]
    if isfloat(ra):
        dla_ra.append(float(ra))
        dla_dec.append(float(dec))
        dla_z.append(float(d))
        qso_z.append(float(q))

dz=.01
rdshift = np.arange(0, 7.2, dz)
n_dla = stats.gaussian_kde(dla_z,bw_method=0.1)(rdshift)
n_qso = stats.gaussian_kde(qso_z,bw_method=0.1)(rdshift)

parameters = ccl.Parameters(Omega_c=0.27,Omega_b=0.045,h=0.69,sigma8=0.83,n_s=0.96)
cosmo = ccl.Cosmology(parameters)
lens = ccl.ClTracerCMBLensing(cosmo)


# In[52]:

b_min_qso = 1; b_min_dla = 1 #starting values for bias_qso and bias_dla

b_min = 0; b_max = 10; db = .01 #parameters for array of allowed biases #################################

bias_qso = b_min_qso*np.ones(rdshift.size); b_qso_fiducial = b_min_qso*np.ones(ell_arr.size)
bias_dla = b_min_dla*np.ones(rdshift.size); b_dla_fiducial = b_min_dla*np.ones(ell_arr.size)

source_dla = ccl.ClTracerNumberCounts(cosmo, False, False, z=rdshift, n=n_dla, bias= bias_dla)
theory_dla = ccl.angular_cl(cosmo,lens,source_dla,ell_arr,l_limber=-1)

source_qso = ccl.ClTracerNumberCounts(cosmo, False, False, z=rdshift, n=n_qso, bias= bias_qso)
theory_qso = ccl.angular_cl(cosmo,lens,source_qso,ell_arr,l_limber=-1)


# In[ ]:

def MCMC():
    slist=[]
    pos=np.zeros(Nd)
    clike=loglike_tot(pos)
    while len(slist)<1e5:
        prop=abs(pos+np.random.normal(0,0.5,Nd)) ## proposal width 0.5
        plike=loglike_tot(prop)
        accept = np.random.uniform(0,1.0)<np.exp(plike-clike)
        if accept:
            pos=prop
            clike=plike
        slist.append(pos)
    return np.array(slist)[1000:] ## cut first 1000 for burn in

def loglike(x):
    theory_prop = np.array([theory[0]*(x[0]/b_min_qso),theory[0]*(x[0]/b_min_qso)+theory[1]*(x[1]/b_min_dla)])
    d = theory_prop - data
    chi2= np.dot(d,np.dot(iCov,np.transpose(d)))
    return -chi2/2.0

def loglike_tot(x):
    if np.amin(x) >= b_min and np.amax(x) < b_max:
        return -chi2_tot[int((x[0]-b_min)/db),int((x[1]-b_min)/db)]/2
    else:
        return -float('inf')


# In[ ]:

Nd=2
Cov = np.zeros((2,2))
Delta = np.zeros((2,1))
dof = 0

Chi_SQ = []; mean = np.zeros((len(ell_arr),2)); error = np.zeros((len(ell_arr),2)); freedom = []; prob = []
chi2_tot = np.zeros(((b_max-b_min)/db,(b_max-b_min)/db))


for ell_bin in range(len(ell_arr)):
    theory = np.array([theory_qso[ell_bin], theory_dla[ell_bin]])
    data = np.array([data_qso[ell_bin],data_dla[ell_bin]])

    Cov[0,0] = np.average(sim_qso[ell_bin,:] * sim_qso[ell_bin,:])
    Cov[1,0] = np.average(sim_dla[ell_bin,:] * sim_qso[ell_bin,:])
    Cov[0,1] = np.average(sim_qso[ell_bin,:] * sim_dla[ell_bin,:])
    Cov[1,1] = np.average(sim_dla[ell_bin,:] * sim_dla[ell_bin,:])
    iCov = np.array(la.inv(Cov))
    
    for i in range(len(chi2_tot)):
        for j in range(len(chi2_tot)):
            biases = np.array([b_min+i*db,b_min+j*db])
            chi2_tot[i,j] += -2*loglike(biases)
    
    trace = MCMC()
    mean[ell_bin,:] = np.average(trace,axis=0)
    error[ell_bin,:] = np.std(trace,axis=0)  
    
    dof += 2
    Chi_SQ.append((chi2_tot[int((mean[ell_bin,0]-b_min)/db),int((mean[ell_bin,1]-b_min)/db)]))
    freedom.append(dof)
    prob.append(stats.chi2.cdf(Chi_SQ[ell_bin],dof))

    
bias = {'ell_max':ell_arr,'b_qso':mean[:,0],'b_dla':mean[:,1],'b_qso_error':error[:,0],'b_dla_error':error[:,1],'chi2':np.array(Chi_SQ),'dof':np.array(freedom),'prob':np.array(prob)}
pickle.dump(bias,open('/Users/Joseph/Brookhaven/Research/pickled/bias.pkl','wb'))


# In[ ]:

for ell_bin in range(len(ell_arr)):
    print('max_ell: '+str(int(ell_arr[ell_bin]))+'   b_qso: '+str(np.round(mean[ell_bin,0],2))+' ± '+str(np.round(error[ell_bin,0],2))+'   b_dla: '+str(np.round(mean[ell_bin,1],2))+' ± '+str(np.round(error[ell_bin,1],2))+'   Chi2: '+str(np.round(Chi_SQ[ell_bin],2))+'    prob: '+str(np.round(stats.chi2.cdf(Chi_SQ[ell_bin],2*ell_bin+2),3)))


# In[ ]:

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.imshow(chi2_tot, interpolation='nearest')
plt.colorbar()
plt.show()


# In[ ]:




# In[ ]:



