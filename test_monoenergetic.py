import monkes
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
import os
#import jax

import jax.numpy as jnp
import numpy as np
import h5py as h5
# Use 8 CPU devices
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=1'


#Species information is not necessary for 'monoenergetic like mode'

#We need the magnetic configuration
nt = 25
nz = 25
eq='wout_QI_nfp2_initial_hires.nc'


#Field at rho=0.12247, s= 0.0149989009
field = monkes.Field.from_vmec(eq, 0.0149989009, nt, nz)
#Field at rho=0.5, s= 0.25
#field = monkes.Field.from_vmec(eq, 0.25, nt, nz)

#nu_hat used for rho=0.5
#nu_v_list=[1.e-6,3.e-6,1.e-5,3.e-5,1.e-4,3.e-4,1.e-3,3.e-3,1.e-2,3.e-2,1.e-1,3.e-1,1.e+0,3.e+0,1.e+1]
#nu_hat used for rho=0.12247
nu_v_list=[3.e-7,1.e-6,3.e-6,1.e-5,3.e-5,1.e-4,3.e-4,1.e-3,3.e-3,1.e-2,3.e-2,1.e-1,3.e-1,1.e+0,3.e+0,1.e+1]
nu_v=np.array(nu_v_list)

#Ehat_rtilde, with rtilde the IPP DKES one
Er_list=[0.0,3.e-6,1.e-5,3.e-5,1.e-4,3.e-4,1.e-3,3.e-3,1.e-2,3.e-2]
#Eshat values, VMEC is in s coordinate, so we need to multiply by ds/dr_tilde for the correct Eshat values
#dsdr_tilde at rho=0.12247
dsdr_tilde=-49.69813723106871
#dsdr_tilde at rho=0.5 
#dsdr_tilde=-12.08953464758394
Er=dsdr_tilde*np.array(Er_list)

#Create arrays for Dij's monoenergetic scan data 
D11=np.zeros((len(nu_v),len(Er)))
D13=np.zeros((len(nu_v),len(Er)))
D31=np.zeros((len(nu_v),len(Er)))
D33=np.zeros((len(nu_v),len(Er)))


#Loop for every collisionality and electric field value to obtain the monoenergetic scan (use mode=3 for monoenergetic like scan, v=1. in this case and we add nu_v)

for j in range(len(nu_v)):
    for i in range(len(Er)):        
        Dij, f, s = monkes.monoenergetic_dke_solve_internal(field, nl=100, Erhat=Er[i],nuhat=nu_v[j])
        D11[j,i]=Dij[0,0]
        D13[j,i]=Dij[0,2]
        D31[j,i]=Dij[2,0]
        D33[j,i]=Dij[2,2]    
        print(Dij)
        print(j)




#Write data in hdf5 file
file=h5.File('Dij.h5','w')
file['D11']=D11
file['D13']=D13
file['D31']=D31
file['D33']=D33
file.close()
print('Ended')

