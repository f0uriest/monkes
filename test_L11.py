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

nt = 25
nz = 31
Nx=4


#Load magnetic configuration using VMEC
eq='wout_QI_nfp2_initial_hires.nc'
#Calculate field at rho=0.29
field = monkes.Field.from_vmec(eq, 0.29**2, nt, nz)

#Create x=v/vth grid and weights for quadrature of integral (using sfincs like method in jax)
grid = monkes.xGrid.sfincsGrid(Nx=Nx,pointAtX0=False)
x=grid.abscissae
xWeights=grid.xWeights

#Now generate electron and ions species
ne = 4.21e20
te = 17.8e+3
ni = 4.21e20
ti = 17.8e+3


#Species calculated at rho
electrons = monkes.GlobalMaxwellian(monkes.Electron, lambda r: te*(1-r**2), lambda r: ne*(1-r**10.))
ions = monkes.GlobalMaxwellian(monkes.Deuterium, lambda r: ti*(1-r**2), lambda r: ni*(1-r**10.))

#SFINCS non-mono/mono uses lorentz operator only with same species collisions, thus sue the same species here to calculate collisionality values
species_ions = [ions, ions]
species_electrons = [electrons, ions]

#Retrieve thermal vwewlocities at radial coordinate of interest to get
vth_ion=ions.v_thermal(r=0.29)
vth_electron=electrons.v_thermal(r=0.29)
#Get v grid for ions and electron to get Dij, based on x grid
v_ion=x*vth_ion
v_electron=x*vth_electron

#Define the electric field to use (checking this to match sfincs!!!!!! this needs to be checked)
Er=17.8e+3#*(2./1.19*0.29)

D11i=np.zeros(len(v_ion))
D11e=np.zeros(len(v_electron))

L11i_fac=ions.density(r=0.29)*2./jnp.sqrt(jnp.pi)*jnp.square(ions.species.mass/ions.species.charge)#*2.*ions.temperature(r=0.29)*1.6e-19/ions.species.mass
L11e_fac=electrons.density(r=0.29)*2./jnp.sqrt(jnp.pi)*jnp.square(electrons.species.mass/jnp.abs(electrons.species.charge))#*2.*electrons.temperature(r=0.29)*1.6e-19/electrons.species.mass
L11i_weight=x*x*x*v_ion*v_ion
L11e_weight=x*x*x*v_electron*v_electron

#Calculate Dij over the x grid for ions and electrons
for j in range(len(v_ion)):       
    Dij_ions, f, s = monkes.monoenergetic_dke_solve(field, species_ions, Er, v=v_ion[j], nl=32)
    D11i[j]=Dij_ions[0,0]
    Dij_electrons, f, s = monkes.monoenergetic_dke_solve(field, species_electrons, Er, v=v_electron[j], nl=32)
    D11e[j]=Dij_electrons[0,0]
    print(j)
    print(Dij_ions)
    print(Dij_electrons)

#Now to calculate L11 we just need to use the xWeights 
L11i=L11i_fac*jnp.sum(L11i_weight*xWeights*D11i)
L11e=L11e_fac*jnp.sum(L11e_weight*xWeights*D11e)

print('L11e',L11e)
print('L11i',L11i)


#file=h5.File('Dij.h5','w')
#file['DL']=D11
#file.close()
#print('Ended')

