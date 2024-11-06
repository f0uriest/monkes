import monkes
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
import os
import jax.numpy as jnp
import numpy as np
import h5py as h5
from netCDF4 import Dataset
import interpax
import matplotlib.pyplot as plt


#Resolution parameters
nt = 25
nz = 25
#Pitch angle resolution
nl=100  

#eq = 'wout_hydra_Np04_20190108.fix001.03.890_12x12.nc'
#eq='wout_QI_nfp2_initial_hires.nc'
#
#Setting VMEC equilibrium and boozer file (needed for correct conversion factors)
eq='wout_QI_nfp2_newNT_opt_hires.nc'
booz='boozermn_wout_QI_nfp2_newNT_opt_hires.nc'
#eq='wout_hydra_Np04_20190108.fix001.03.890_12x12.nc'
#booz='boozermn_wout_hydra_Np04_20190108.fix001.03.890_12x12.nc'
vmec=eq

#Typical rho, nu/v and E_rtilde/(v B0) values used in DKES IPP databases 
rho_list=[0.12247,0.25,0.375,0.5,0.625,0.75,0.875]
nu_v_list=[3.e-7,1.e-6,3.e-6,1.e-5,3.e-5,1.e-4,3.e-4,1.e-3,3.e-3,1.e-2,3.e-2,1.e-1,3.e-1,1.e+0,3.e+0,1.e+1]
Er_list=[0.0,3.e-6,1.e-5,3.e-5,1.e-4,3.e-4,1.e-3,3.e-3,1.e-2,3.e-2,1.e-1]

#Convert to 
nu_v=jnp.array(nu_v_list)
rho=jnp.array(rho_list)
Er_tilde=np.array(Er_list)

#Read files outside of MONKES-JAX for some auxiliary quantities for conversion
vfile = Dataset(vmec, mode="r")
bfile = Dataset(booz, mode="r")

ns = vfile.variables["ns"][:].filled()
s_full = jnp.linspace(0,1,ns)  #This is s_full
s_half_list = [(i-0.5)/(ns-1) for i in range(0,ns)] #This is s_half
s_half =jnp.array(s_half_list)

rho_half=jnp.sqrt(s_half)
rho_full=jnp.sqrt(s_full)

Vprime = vfile.variables["vp"][:].filled()
Aminor_p = vfile.variables["Aminor_p"][:].filled()   
volume_p = vfile.variables["volume_p"][:].filled()
vp = vfile.variables["vp"][:].filled()  
phi = vfile.variables["phi"][:].filled()  
iotaf = vfile.variables["iotaf"][:].filled()
phipf = vfile.variables["phipf"][:].filled()                                
Psia=jnp.abs(phi[-1])/(2.*jnp.pi)
vfile.close()
bmnc_b=bfile.variables["bmnc_b"][:].filled() 
rmnc_b=bfile.variables["rmnc_b"][:].filled()
gmnc_b=bfile.variables['gmn_b'][:].filled()
xm_b=bfile.variables['ixm_b'][:].filled()
xn_b=bfile.variables['ixn_b'][:].filled()
buco=bfile.variables['buco_b'][:].filled()
bvco=bfile.variables['bvco_b'][:].filled()
bfile.close()
R0_b=rmnc_b[-1,0]
a_b=np.sqrt(volume_p/(2*np.pi**2*R0_b))  


for l in range(len(xm_b)):
    if(xm_b[l]==0 and xn_b[l]==0):
        B00=interpax.Interpolator1D(rho_half[1:],bmnc_b[:,l],extrap=True)
        R00=interpax.Interpolator1D(rho_full[1:],rmnc_b[:,l],extrap=True)
    if(xm_b[l]==1 and xn_b[l]==0):
        B10=interpax.Interpolator1D(rho_half[1:],bmnc_b[:,l],extrap=True)

dVdr=interpax.Interpolator1D(rho_half[1:],vp[1:],extrap=True)
I=interpax.Interpolator1D(rho_half[1:],buco[1:],extrap=True)
G=interpax.Interpolator1D(rho_half[1:],bvco[1:],extrap=True)
iota=interpax.Interpolator1D(rho_full[:],iotaf[:],extrap=True)

B00_rho=B00(rho)
R00_rho=R00(rho)
I_rho=I(rho)
G_rho=G(rho)
iota_rho=iota(rho)
dPsidrtilde=rho*a_b*B00_rho
drds=a_b/(rho*2.)
dr_tildedr=2.*Psia/(a_b**2*B00_rho)
dr_tildeds=dr_tildedr*drds

#Ceate arrays for different electric field representations
Es=np.zeros((len(rho),len(Er_tilde)))
Er=np.zeros((len(rho),len(Er_tilde)))
Er_to_Ertilde=np.zeros((len(rho),len(Er_tilde)))   #Er to E_rtilde conversion factor


#Factor to convert from MONKES in VMEC 'r' coordinate to SFINCS Lij (whin are in psi coordinate for mono energetic case), see M. Landreman notes on this
Fac_MONKES_TO_SFINCS_11=(8.*(G_rho+iota_rho*I_rho)*B00_rho/(jnp.sqrt(jnp.pi)*G_rho**2))
Fac_MONKES_TO_SFINCS_31=(4.*B00_rho/(np.sqrt(jnp.pi)*G_rho))
Fac_MONKES_TO_SFINCS_33=1. #tODO

#Factor to convert from SFINCS  in VMEC 'psi' coordinate to Gamma_hat in DKES IPP which uses r_tilde coordinate, see H. Smith notes on this, or appendix of Electron root paper
Fac_SFINCS_TO_DKES_11=1./(8.*(G_rho+iota_rho*I_rho)/(G_rho**2*B00_rho*jnp.sqrt(jnp.pi))*dPsidrtilde**2)
Fac_SFINCS_TO_DKES_31=1./(4./(G_rho*jnp.sqrt(jnp.pi))*dPsidrtilde)
Fac_SFINCS_TO_DKES_33=1./(-2./(G_rho+iota_rho*I_rho*jnp.sqrt(jnp.pi))*B00_rho)

#Factor to convert from Gamma_hat IPP to Dij_stars of C. Beidler (These are the coefficients normalised to the equivalent tokamak/banana-like regime)
Fac_DKES_TO_D11star=-8/jnp.pi*iota_rho*R00_rho*jnp.square(B00_rho)/jnp.square(B00_rho)  #The squared B00 in denominator is needed here because of B00=1 in DKES IPP files
epsilon_t=rho*a_b/R00_rho
Fac_DKES_TO_D31star=-3./1.46*iota_rho*jnp.sqrt(epsilon_t)/2.
Fac_DKES_TO_D33star=1.#todo



#SFINCS results in DKES IPP Gamma_hat format for benchmark, using an electron root optimised equilibrium
sfincs1=h5.File('DKES_rho_0.12247.h5','r')
Gamma11_0p1=sfincs1['Gamma11'][()]
Estar_0p1_Opt=sfincs1['xEr'][()][0]
sfincs1.close()

sfincs1=h5.File('DKES_rho_0.25.h5','r')
Gamma11_0p2=sfincs1['Gamma11'][()]
Estar_0p2_Opt=sfincs1['xEr'][()][0]
sfincs1.close()

sfincs1=h5.File('DKES_rho_0.375.h5','r')
Gamma11_0p3=sfincs1['Gamma11'][()]
Estar_0p3_Opt=sfincs1['xEr'][()][0]
sfincs1.close()

sfincs1=h5.File('DKES_rho_0.5.h5','r')
Gamma11_0p5=sfincs1['Gamma11'][()]
Estar_0p5_Opt=sfincs1['xEr'][()][0]
sfincs1.close()

sfincs1=h5.File('DKES_rho_0.625.h5','r')
Gamma11_0p6=sfincs1['Gamma11'][()]
Estar_0p6_Opt=sfincs1['xEr'][()][0]
sfincs1.close()

sfincs1=h5.File('DKES_rho_0.75.h5','r')
Gamma11_0p7=sfincs1['Gamma11'][()]
Estar_0p7_Opt=sfincs1['xEr'][()][0]
sfincs1.close()

sfincs1=h5.File('DKES_rho_0.875.h5','r')
Gamma11_0p8=sfincs1['Gamma11'][()]
Estar_0p8_Opt=sfincs1['xEr'][()][0]
sfincs1.close()


#Create arrays for Dij's monoenergetic scan data 
D11=np.zeros((len(rho),len(nu_v),len(Er_tilde)))
D13=np.zeros((len(rho),len(nu_v),len(Er_tilde)))
D31=np.zeros((len(rho),len(nu_v),len(Er_tilde)))
D33=np.zeros((len(rho),len(nu_v),len(Er_tilde)))


#Loop for every collisionality and electric field value to obtain the monoenergetic scan 
#Use internal solve as we do not care for species information in the monoenergetic database
#Using normal for loop here as it serves only for benchmark-> put into vmap to speed up in real calculations
for si in range(len(rho)):
    field = monkes.Field.from_vmec(eq, rho[si]**2, nt, nz)     
    for j in range(len(nu_v)):       
        for i in range(len(Er_tilde)):
            #Here we use input 
            Es[si,i]=Er_tilde[i]*dr_tildeds[si]*B00_rho[si] #Notice we need to multiply by B00 and dr_tildeds factors due to IPP DKES weird coordinates
            Er[si,i]=Er_tilde[i]*dr_tildedr[si]*B00_rho[si]
            Er_to_Ertilde[si,i]=1./dr_tildedr[si]
            #Calculate one Dij matrix 
            Dij, f, s = monkes._core.monoenergetic_dke_solve_internal(field, nl=nl, Erhat=Er[si,i],nuhat=nu_v[j])
            D11[si,j,i]=Dij[0,0]
            D13[si,j,i]=Dij[0,2]
            D31[si,j,i]=Dij[2,0]
            D33[si,j,i]=Dij[2,2]    
            print(si,j,i)
            print(Dij)


#Write data in hdf5 file
file=h5.File('Dij_BENCHMARK_r.h5','w')
file['rho']=rho
file['nu_v']=nu_v
file['Er']=Er
file['Er_tilde']=Er_tilde
file['Er_to_Ertilde']=Er_to_Ertilde
file['Es']=Es
file['drds']=drds
file['dr_tildedr']=dr_tildedr
file['dr_tildeds']=dr_tildeds
file['D11']=D11
file['D13']=D13
file['D31']=D31
file['D33']=D33
file['Fac_MONKES_TO_SFINCS_11']=Fac_MONKES_TO_SFINCS_11
file['Fac_MONKES_TO_SFINCS_31']=Fac_MONKES_TO_SFINCS_31
file['Fac_MONKES_TO_SFINCS_33']=Fac_MONKES_TO_SFINCS_33
file['Fac_SFINCS_TO_DKES_11']=Fac_SFINCS_TO_DKES_11
file['Fac_SFINCS_TO_DKES_31']=Fac_SFINCS_TO_DKES_31
file['Fac_SFINCS_TO_DKES_33']=Fac_SFINCS_TO_DKES_33
file['Fac_DKES_TO_D11star']=Fac_DKES_TO_D11star
file['Fac_DKES_TO_D31star']=Fac_DKES_TO_D31star
file['Fac_DKES_TO_D33star']=Fac_DKES_TO_D33star
file.close()
print('Ended')





#Read the file again for ploting the benchmarks
Opt=h5.File('Dij_BENCHMARK_r.h5','r')
nu_v=Opt['nu_v'][()]
D11_Opt_DKES=Opt['D11'][()]
D31_Opt_DKES=Opt['D31'][()]
Fac_MONKES_TO_SFINCS_11_OPT=Opt['Fac_MONKES_TO_SFINCS_11'][()]
Fac_MONKES_TO_SFINCS_31_OPT=Opt['Fac_MONKES_TO_SFINCS_31'][()]
Fac_MONKES_TO_SFINCS_33_OPT=Opt['Fac_MONKES_TO_SFINCS_33'][()]
Fac_SFINCS_TO_DKES_11_OPT=Opt['Fac_SFINCS_TO_DKES_11'][()]
Fac_SFINCS_TO_DKES_31_OPT=Opt['Fac_SFINCS_TO_DKES_31'][()]
Fac_SFINCS_TO_DKES_33_OPT=Opt['Fac_SFINCS_TO_DKES_33'][()]
Fac_DKES_TO_D11star_OPT=Opt['Fac_DKES_TO_D11star'][()]
Fac_DKES_TO_D31star_OPT=Opt['Fac_DKES_TO_D31star'][()]
Fac_DKES_TO_D33star_OPT=Opt['Fac_DKES_TO_D33star'][()]
Opt.close()


fig,ax=plt.subplots(dpi=120)
ax.title('$r/a=0.12247$')
ax.loglog(nu_v,-Gamma11_0p1[::-1,:],'o')
ax.loglog(nu_v,D11_Opt_DKES[0,:,:]*Fac_MONKES_TO_SFINCS_11_OPT[0]*Fac_SFINCS_TO_DKES_11_OPT[0],'x')
plt.xlabel('$\\nu/v$')
plt.ylabel('$\\hat{\Gamma}^{IPP_DKES}_{11}$')
plt.savefig('D11_rho_0.12247.pdf', bbox_inches = 'tight', pad_inches = 0)

fig,ax=plt.subplots(dpi=120)
ax.title('$r/a=0.25$')
ax.loglog(nu_v,-Gamma11_0p2[::-1,:],'o')
ax.loglog(nu_v,D11_Opt_DKES[1,:,:]*Fac_MONKES_TO_SFINCS_11_OPT[1]*Fac_SFINCS_TO_DKES_11_OPT[1],'x')
plt.xlabel('$\\nu/v$')
plt.ylabel('$\\hat{\Gamma}^{IPP_DKES}_{11}$')
plt.savefig('D11_rho_0.25.pdf', bbox_inches = 'tight', pad_inches = 0)

fig,ax=plt.subplots(dpi=120)
ax.title('$r/a=0.375$')
ax.loglog(nu_v[1:],-Gamma11_0p3[::-1,:],'o')
ax.loglog(nu_v,D11_Opt_DKES[2,:,:]*Fac_MONKES_TO_SFINCS_11_OPT[2]*Fac_SFINCS_TO_DKES_11_OPT[2],'x')
plt.xlabel('$\\nu/v$')
plt.ylabel('$\\hat{\Gamma}^{IPP_DKES}_{11}$')
plt.savefig('D11_rho_0.375.pdf', bbox_inches = 'tight', pad_inches = 0)

fig,ax=plt.subplots(dpi=120)
ax.title('$r/a=0.5$')
ax.loglog(nu_v[1:],-Gamma11_0p5[::-1,:],'o')
ax.loglog(nu_v,D11_Opt_DKES[3,:,:]*Fac_MONKES_TO_SFINCS_11_OPT[3]*Fac_SFINCS_TO_DKES_11_OPT[3],'x')
plt.xlabel('$\\nu/v$')
plt.ylabel('$\\hat{\Gamma}^{IPP_DKES}_{11}$')
plt.savefig('D11_rho_0.5.pdf', bbox_inches = 'tight', pad_inches = 0)

fig,ax=plt.subplots(dpi=120)
ax.title('$r/a=0.625$')
ax.loglog(nu_v[1:],-Gamma11_0p6[::-1,:],'o')
ax.loglog(nu_v,D11_Opt_DKES[4,:,:]*Fac_MONKES_TO_SFINCS_11_OPT[4]*Fac_SFINCS_TO_DKES_11_OPT[4],'x')
plt.xlabel('$\\nu/v$')
plt.ylabel('$\\hat{\Gamma}^{IPP_DKES}_{11}$')
plt.savefig('D11_rho_0.625.pdf', bbox_inches = 'tight', pad_inches = 0)

fig,ax=plt.subplots(dpi=120)
ax.title('$r/a=0.75$')
ax.loglog(nu_v[1:],-Gamma11_0p7[::-1,:],'o')
ax.loglog(nu_v,D11_Opt_DKES[5,:,:]*Fac_MONKES_TO_SFINCS_11_OPT[5]*Fac_SFINCS_TO_DKES_11_OPT[5],'x')
plt.xlabel('$\\nu/v$')
plt.ylabel('$\\hat{\Gamma}^{IPP_DKES}_{11}$')
plt.savefig('D11_rho_0.75.pdf', bbox_inches = 'tight', pad_inches = 0)

fig,ax=plt.subplots(dpi=120)
ax.title('$r/a=0.875$')
ax.loglog(nu_v[1:],-Gamma11_0p8[::-1,:],'o')
ax.loglog(nu_v,D11_Opt_DKES[6,:,:]*Fac_MONKES_TO_SFINCS_11_OPT[6]*Fac_SFINCS_TO_DKES_11_OPT[6],'x')
plt.xlabel('$\\nu/v$')
plt.ylabel('$\\hat{\Gamma}^{IPP_DKES}_{11}$')
plt.savefig('D11_rho_0.875.pdf', bbox_inches = 'tight', pad_inches = 0)