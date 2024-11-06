import functools
import warnings

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jaxtyping import Array, Float


class Field(eqx.Module):
    """Magnetic field on a flux surface.

    Field is given as 2D arrays uniformly spaced in arbitrary
    poloidal and toroidal angles.

    Parameters
    ----------
    rho : float
        Flux surface label.
    B_sup_t : jax.Array, shape(ntheta, nzeta)
        B^theta, contravariant poloidal component of field.
    B_sup_z : jax.Array, shape(ntheta, nzeta)
        B^zeta, contravariant toroidal component of field.
    B_sub_t : jax.Array, shape(ntheta, nzeta)
        B_theta, covariant poloidal component of field.
    B_sub_z : jax.Array, shape(ntheta, nzeta)
        B_zeta, covariant toroidal component of field.
    Bmag : jax.Array, shape(ntheta, nzeta)
        Magnetic field magnitude.
    sqrtg : jax.Array, shape(ntheta, nzeta)
        Coordinate jacobian determinant from (psi, theta, zeta) to (R, phi, Z).
    psi_r : float
        Derivative of toroidal flux wrt minor radius (rho*a_minor)
    NFP : int
        Number of field periods.
    deriv_mode : {"fft", "fd2", "fd4", "fd6"}
        Method to use for approximating poloidal and toroidal derivatives. "fft" uses
        spectral differentiation of the Fourier series, fd{2,4,6} uses centered finite
        differences of the specified order.
    """

    # note: assumes (psi, theta, zeta) coordinates, not (rho, theta, zeta)
    rho: float
    theta: Float[Array, "ntheta "]
    zeta: Float[Array, "nzeta "]
    wtheta: Float[Array, "ntheta "]
    wzeta: Float[Array, "nzeta "]
    B_sup_t: Float[Array, "ntheta nzeta"]
    B_sup_z: Float[Array, "ntheta nzeta"]
    B_sub_t: Float[Array, "ntheta nzeta"]
    B_sub_z: Float[Array, "ntheta nzeta"]
    sqrtg: Float[Array, "ntheta nzeta"]
    Bmag: Float[Array, "ntheta nzeta"]
    bdotgradB: Float[Array, "ntheta nzeta"]
    BxgradpsidotgradB: Float[Array, "ntheta nzeta"]
    dBdt: Float[Array, "ntheta nzeta"]
    dBdz: Float[Array, "ntheta nzeta"]
    Bmag_fsa: float
    B2mag_fsa: float
    psi_r: float
    iota: float
    B0: float
    ntheta: int = eqx.field(static=True)
    nzeta: int = eqx.field(static=True)
    NFP: int = eqx.field(static=True)
    deriv_mode: str = eqx.field(static=True)

    def __init__(
        self,
        rho: float,
        B_sup_t: Float[Array, "ntheta nzeta"],
        B_sup_z: Float[Array, "ntheta nzeta"],
        B_sub_t: Float[Array, "ntheta nzeta"],
        B_sub_z: Float[Array, "ntheta nzeta"],
        Bmag: Float[Array, "ntheta nzeta"],
        sqrtg: Float[Array, "ntheta nzeta"],
        psi_r: float,
        iota: float,
        NFP: int = 1,
        *,
        deriv_mode: str = "fft",
        dBdt=None,
        dBdz=None,
        B0=None,
    ):
        assert deriv_mode in ["fft", "fd2", "fd4", "fd6"]
        self.deriv_mode = deriv_mode
        self.rho = rho
        self.NFP = NFP
        self.ntheta = sqrtg.shape[0]
        self.nzeta = sqrtg.shape[1]
        assert (self.ntheta % 2 == 1) and (
            self.nzeta % 2 == 1
        ), "ntheta and nzeta must be odd"
        if "fd" in deriv_mode:
            min_length = int(deriv_mode[-1]) + 1
            assert self.ntheta >= min_length and self.nzeta >= min_length
        self.B_sup_t = B_sup_t
        self.B_sup_z = B_sup_z
        self.B_sub_t = B_sub_t
        self.B_sub_z = B_sub_z
        self.sqrtg = sqrtg
        self.Bmag = Bmag
        if dBdt is None:
            dBdt = self._dfdt(self.Bmag)
        if dBdz is None:
            dBdz = self._dfdz(self.Bmag)
        self.dBdt = dBdt
        self.dBdz = dBdz
        if B0 is None:
            B0 = Bmag.mean()
        self.B0 = B0
        self.bdotgradB = (B_sup_t * dBdt + B_sup_z * dBdz) / self.Bmag
        self.BxgradpsidotgradB = (B_sub_z * dBdt - B_sub_t * dBdz) / sqrtg
        self.Bmag_fsa = self.flux_surface_average(self.Bmag)
        self.B2mag_fsa = self.flux_surface_average(self.Bmag**2)
        self.psi_r = psi_r
        self.iota = iota
        self.theta = jnp.linspace(0, 2 * np.pi, self.ntheta, endpoint=False)
        self.zeta = jnp.linspace(0, 2 * np.pi / NFP, self.nzeta, endpoint=False)
        self.wtheta = jnp.diff(self.theta, append=jnp.array([2 * jnp.pi]))
        self.wzeta = jnp.diff(self.zeta, append=jnp.array([2 * jnp.pi / NFP]))

    @classmethod
    def from_desc(
        cls, eq, rho: int, ntheta: float, nzeta: float, deriv_mode: str = "fft"
    ):
        """Construct Field from DESC equilibrium.

        Parameters
        ----------
        eq : desc.equilibrium.Equilibrium
            DESC Equilibrium.
        rho : float
            Flux surface label.
        ntheta, nzeta : int
            Number of points on a surface in poloidal and toroidal directions.
            Both must be odd.
        deriv_mode : {"fft", "fd2", "fd4", "fd6"}
            Method to use for approximating poloidal and toroidal derivatives.
            "fft" uses spectral differentiation of the Fourier series, fd{2,4,6} uses
            centered finite differences of the specified order.
        """
        assert (ntheta % 2 == 1) and (nzeta % 2 == 1), "ntheta and nzeta must be odd"

        from desc.grid import LinearGrid

        grid = LinearGrid(rho=rho, theta=ntheta, zeta=nzeta, endpoint=False, NFP=eq.NFP)
        keys = [
            "B^theta",
            "B^zeta",
            "B_theta",
            "B_zeta",
            "|B|",
            "|B|_t",
            "|B|_z",
            "sqrt(g)",
            "psi_r",
            "iota",
            "a",
        ]
        desc_data = eq.compute(keys, grid=grid)

        data = {
            "B_sup_t": desc_data["B^theta"],
            "B_sup_z": desc_data["B^zeta"],
            "B_sub_t": desc_data["B_theta"],
            "B_sub_z": desc_data["B_zeta"],
            "Bmag": desc_data["|B|"],
            "dBdt": desc_data["|B|_t"],
            "dBdz": desc_data["|B|_z"],
            "sqrtg": desc_data["sqrt(g)"] / desc_data["psi_r"],
        }

        data = {
            key: val.reshape((grid.num_theta, grid.num_zeta), order="F")
            for key, val in data.items()
        }
        return cls(
            rho=rho,
            psi_r=desc_data["psi_r"][0] / desc_data["a"],
            iota=desc_data["iota"][0],
            **data,
            NFP=eq.NFP,
            deriv_mode=deriv_mode,
        )

    @classmethod
    def from_vmec_desc(cls, wout, s: float, ntheta: int, nzeta: int):
        """Construct Field from VMEC equilibrium.


        Parameters
        ----------
        wout : path-like
            Path to vmec wout file.
        s : float
            Flux surface label.
        ntheta, nzeta : int
            Number of points on a surface in poloidal and toroidal directions.
        """

        assert (ntheta % 2 == 1) and (nzeta % 2 == 1), "ntheta and nzeta must be odd"

        from desc.grid import LinearGrid
        from desc.vmec import VMECIO

        rho=jnp.sqrt(s)
        eq=VMECIO.load(wout)

        grid = LinearGrid(rho=rho, theta=ntheta, zeta=nzeta, endpoint=False, NFP=eq.NFP)
        keys = [
            "B^theta",
            "B^zeta",
            "B_theta",
            "B_zeta",
            "|B|",
            "sqrt(g)",
            "psi_r",
            "a",
        ]
        desc_data = eq.compute(keys, grid=grid)

        data = {
            "B_sup_t": desc_data["B^theta"],
            "B_sup_z": desc_data["B^zeta"],
            "B_sub_t": desc_data["B_theta"],
            "B_sub_z": desc_data["B_zeta"],
            "Bmag": desc_data["|B|"],
#            "sqrtg": desc_data["sqrt(g)"] ,
            "sqrtg": desc_data["sqrt(g)"] / desc_data["psi_r"],
 #           "sqrtg": desc_data["sqrt(g)"] / (desc_data["psi_r"]*desc_data["a"]),
        }

        data = {
            key: val.reshape((grid.num_theta, grid.num_zeta), order="F")
            for key, val in data.items()
        }
        return cls(
            rho=rho, psi_r=desc_data["psi_r"][0] / desc_data["a"], **data, NFP=eq.NFP
  #          rho=rho, psi_r=desc_data["psi_r"][0] / desc_data["a"]*desc_data["a"], **data, NFP=eq.NFP
        )

    @classmethod
    def from_vmec(cls,
        vmec,
        s: float,
        ntheta: int,
        nzeta: int,
        cutoff: float = 0.0,
        deriv_mode: str = "fft",
    ):
        """Construct Field from BOOZ_XFORM file.

        Parameters
        ----------
        vmec : path-like
            Path to booz_xform wout file.
        s : float
            Flux surface label.
        ntheta, nzeta : int
            Number of points on a surface in poloidal and toroidal directions.
        """
        assert (ntheta % 2 == 1) and (nzeta % 2 == 1), "ntheta and nzeta must be odd"
        from netCDF4 import Dataset

        file = Dataset(vmec, mode="r")

        ns = file.variables["ns"][:].filled()
        nfp = file.variables["nfp"][:].filled()
        #print('ns',ns)
        #print('nfp',nfp)
        theta = jnp.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        zeta = jnp.linspace(0, 2 * np.pi / nfp, nzeta, endpoint=False)
        assert "bmns" not in file.variables, "non-symmetric booz-xform not supported"

        s_full = jnp.linspace(0, 1, ns)
        hs = 1 / (ns - 1)
        s_half = s_full[0:-1] + hs / 2
        #s_half = jnp.array([(i-0.5)/(ns-1) for i in range(1,ns)])
        #print(s_half.shape)

        volume = file.variables["volume_p"][:].filled()
        Aminor_p = file.variables["Aminor_p"][:].filled()    
        Rmajor_p = file.variables["Rmajor_p"][:].filled() 
        aspect = file.variables["aspect"][:].filled()                   
        r_mnc = file.variables["rmnc"][:].filled()
        g_mnc = file.variables["gmnc"][:].filled()
        b_mnc = file.variables["bmnc"][:].filled()
        bsupu_mnc = file.variables["bsupumnc"][:].filled()
        bsupv_mnc = file.variables["bsupvmnc"][:].filled()
        bsubu_mnc = file.variables["bsubumnc"][:].filled()
        bsubv_mnc = file.variables["bsubvmnc"][:].filled()
        nfp = file.variables["nfp"][:].filled()
        iota = file.variables["iotaf"][:].filled()
        psi_s = file.variables["phipf"][:].filled()
        


        # assuming the field is only over a single flux surface s
        g_mnc = interpax.interp1d(s, s_half, g_mnc[1:,:])
        b_mnc = interpax.interp1d(s, s_half, b_mnc[1:,:])
        bsupu_mnc = interpax.interp1d(s, s_half, bsupu_mnc[1:,:])
        bsupv_mnc = interpax.interp1d(s, s_half, bsupv_mnc[1:,:])
        bsubu_mnc = interpax.interp1d(s, s_half, bsubu_mnc[1:,:])
        bsubv_mnc = interpax.interp1d(s, s_half, bsubv_mnc[1:,:])
        iota = -interpax.interp1d(s, s_full, iota)  # Not necessary but to match boozer we need it here

        psi_s = interpax.interp1d(s, s_full, psi_s)/(2.*jnp.pi)
        #xm = file.variables["xm"][:].filled()
        #xn = file.variables["xn"][:].filled()
        xm = file.variables["xm_nyq"][:].filled()  
        xn = file.variables["xn_nyq"][:].filled()  

        sqrtg = vmec_eval(theta[:, None], zeta[None, :], g_mnc, 0, xm, xn)
        Bmag = vmec_eval(theta[:, None], zeta[None, :], b_mnc, 0, xm, xn)
        B_sub_t = vmec_eval(theta[:, None], zeta[None, :], bsubu_mnc, 0, xm, xn)
        B_sub_z = vmec_eval(theta[:, None], zeta[None, :], bsubv_mnc, 0, xm, xn)                        
        B_sup_t = vmec_eval(theta[:, None], zeta[None, :], bsupu_mnc, 0, xm, xn)
        B_sup_z = vmec_eval(theta[:, None], zeta[None, :], bsupv_mnc, 0, xm, xn)    

        #Copied from recent booz_xform reader changes but need to understand
        B0 = jnp.abs(b_mnc).max()
        mask = jnp.abs(b_mnc) > cutoff * B0

        dBdt = vmec_eval(theta[:, None], zeta[None, :], b_mnc * mask, 0, xm, xn, dt=1)
        dBdz = vmec_eval(theta[:, None], zeta[None, :], b_mnc * mask, 0, xm, xn, dz=1)


        #print(B_sub_t)
        #a_minor = R0 / aspect
        #a_minor=2.01879
        data = {}
        data["sqrtg"] = sqrtg/psi_s
        data["Bmag"] = Bmag
        data["dBdt"] = dBdt
        data["dBdz"] = dBdz
        data["B_sub_t"] = B_sub_t
        data["B_sub_z"] = B_sub_z
        data["B_sup_t"] = B_sup_t*psi_s
        data["B_sup_z"] = B_sup_z*psi_s
        data["psi_r"] = psi_s * 2 * jnp.sqrt(s) / Aminor_p
        data["iota"] = iota
        data["B0"] = B0

        return cls(rho=jnp.sqrt(s), **data, NFP=nfp, deriv_mode=deriv_mode)


    @classmethod
    def from_vmec_s(cls,
        vmec,
        s: float,
        ntheta: int,
        nzeta: int,
        cutoff: float = 0.0,
        deriv_mode: str = "fft",
    ):
        """Construct Field from BOOZ_XFORM file.

        Parameters
        ----------
        vmec : path-like
            Path to booz_xform wout file.
        s : float
            Flux surface label.
        ntheta, nzeta : int
            Number of points on a surface in poloidal and toroidal directions.
        """
        assert (ntheta % 2 == 1) and (nzeta % 2 == 1), "ntheta and nzeta must be odd"
        from netCDF4 import Dataset

        file = Dataset(vmec, mode="r")

        ns = file.variables["ns"][:].filled()
        nfp = file.variables["nfp"][:].filled()
        #print('ns',ns)
        #print('nfp',nfp)
        theta = jnp.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        zeta = jnp.linspace(0, 2 * np.pi / nfp, nzeta, endpoint=False)
        assert "bmns" not in file.variables, "non-symmetric booz-xform not supported"

        s_full = jnp.linspace(0, 1, ns)
        hs = 1 / (ns - 1)
        s_half = s_full[0:-1] + hs / 2
        #s_half = jnp.array([(i-0.5)/(ns-1) for i in range(1,ns)])
        #print(s_half.shape)

        volume = file.variables["volume_p"][:].filled()
        Aminor_p = file.variables["Aminor_p"][:].filled()    
        Rmajor_p = file.variables["Rmajor_p"][:].filled() 
        aspect = file.variables["aspect"][:].filled()                   
        r_mnc = file.variables["rmnc"][:].filled()
        g_mnc = file.variables["gmnc"][:].filled()
        b_mnc = file.variables["bmnc"][:].filled()
        bsupu_mnc = file.variables["bsupumnc"][:].filled()
        bsupv_mnc = file.variables["bsupvmnc"][:].filled()
        bsubu_mnc = file.variables["bsubumnc"][:].filled()
        bsubv_mnc = file.variables["bsubvmnc"][:].filled()
        #print(g_mnc[0,:])
        #print(g_mnc[1,:])
        nfp = file.variables["nfp"][:].filled()
        iota = file.variables["iotaf"][:].filled()
        psi_s = file.variables["phi"][:].filled()
        phi = file.variables["phi"][:].filled()
        #psi_s=psi_s/psi_s[-1]
        #print('phi',phi)

        # assuming the field is only over a single flux surface s
        g_mnc = interpax.interp1d(s, s_half, g_mnc[1:,:])
        b_mnc = interpax.interp1d(s, s_half, b_mnc[1:,:])
        bsupu_mnc = interpax.interp1d(s, s_half, bsupu_mnc[1:,:])
        bsupv_mnc = interpax.interp1d(s, s_half, bsupv_mnc[1:,:])
        bsubu_mnc = interpax.interp1d(s, s_half, bsubu_mnc[1:,:])
        bsubv_mnc = interpax.interp1d(s, s_half, bsubv_mnc[1:,:])
        iota = -interpax.interp1d(s, s_full, iota)  # Not necessary but to match boozer we need it here

        psi_s = interpax.interp1d(s, s_full, psi_s) # Not necessary? but to match boozer we need it here

        #xm = file.variables["xm"][:].filled()
        #xn = file.variables["xn"][:].filled()
        xm = file.variables["xm_nyq"][:].filled()  
        xn = file.variables["xn_nyq"][:].filled()  

        sqrtg = vmec_eval(theta[:, None], zeta[None, :], g_mnc, 0, xm, xn)
        Bmag = vmec_eval(theta[:, None], zeta[None, :], b_mnc, 0, xm, xn)
        B_sub_t = vmec_eval(theta[:, None], zeta[None, :], bsubu_mnc, 0, xm, xn)
        B_sub_z = vmec_eval(theta[:, None], zeta[None, :], bsubv_mnc, 0, xm, xn)                        
        B_sup_t = vmec_eval(theta[:, None], zeta[None, :], bsupu_mnc, 0, xm, xn)
        B_sup_z = vmec_eval(theta[:, None], zeta[None, :], bsupv_mnc, 0, xm, xn)    

        #Copied from recent booz_xform reader changes but need to understand
        B0 = jnp.abs(b_mnc).max()
        mask = jnp.abs(b_mnc) > cutoff * B0

        dBdt = vmec_eval(theta[:, None], zeta[None, :], b_mnc * mask, 0, xm, xn, dt=1)
        dBdz = vmec_eval(theta[:, None], zeta[None, :], b_mnc * mask, 0, xm, xn, dz=1)


        #print(B_sub_t)
        #a_minor = R0 / aspect
        #a_minor=2.01879
        data = {}
        data["sqrtg"] = sqrtg#*phi[-1]
        data["Bmag"] = Bmag
        data["dBdt"] = dBdt
        data["dBdz"] = dBdz
        data["B_sub_t"] = B_sub_t
        data["B_sub_z"] = B_sub_z
        data["B_sup_t"] = B_sup_t
        data["B_sup_z"] = B_sup_z
        data["psi_r"] = 1.#phi[-1] * 2 * jnp.sqrt(s) / a_minor
        data["iota"] = iota
        data["B0"] = B0

        return cls(rho=jnp.sqrt(s), **data, NFP=nfp, deriv_mode=deriv_mode)


    @classmethod
    def from_booz_xform(
        cls,
        booz,
        s: float,
        ntheta: int,
        nzeta: int,
        cutoff: float = 0.0,
        deriv_mode: str = "fft",
    ):
        """Construct Field from BOOZ_XFORM file.

        Parameters
        ----------
        booz : path-like
            Path to booz_xform wout file.
        s : float
            Flux surface label.
        ntheta, nzeta : int
            Number of points on a surface in poloidal and toroidal directions.
        cutoff : float
            Modes with abs(b_mn) < cutoff * abs(b_00) will be excluded.
        deriv_mode : {"fft", "fd2", "fd4", "fd6"}
            Method to use for approximating poloidal and toroidal derivatives.
            "fft" uses spectral differentiation of the Fourier series, fd{2,4,6} uses
            centered finite differences of the specified order.
        """
        assert (ntheta % 2 == 1) and (nzeta % 2 == 1), "ntheta and nzeta must be odd"
        from netCDF4 import Dataset

        file = Dataset(booz, mode="r")
        assert not bool(
            file.variables["lasym__logical__"][:].filled()
        ), "non-symmetric booz-xform not supported"

        ns = file.variables["ns_b"][:].filled()
        nfp = file.variables["nfp_b"][:].filled()

        theta = jnp.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        zeta = jnp.linspace(0, 2 * np.pi / nfp, nzeta, endpoint=False)

        s_full = jnp.linspace(0, 1, ns)

        aspect = file.variables["aspect_b"][:].filled()
        b_mnc = file.variables["bmnc_b"][:].filled()
        r_mnc = file.variables["rmnc_b"][:].filled()
        nfp = file.variables["nfp_b"][:].filled()
        iota = file.variables["iota_b"][:].filled()
        psi_s = file.variables["phip_b"][:].filled()
        buco = file.variables["buco_b"][:].filled()  # (AKA Boozer I)
        bvco = file.variables["bvco_b"][:].filled()  # (AKA Boozer G)

        # copied from fortran monkes, need to understand this
        R0 = r_mnc[1, 0]
        a_minor = R0 / aspect

        # assuming the field is only over a single flux surface s
        # bmnc on full mesh? this seems to agree with fortran
        b_mnc = interpax.interp1d(s, s_full[1:], b_mnc)
        buco = -interpax.interp1d(s, s_full, buco)  # sign flip LH -> RH
        bvco = interpax.interp1d(s, s_full, bvco)
        iota = -interpax.interp1d(s, s_full, iota)  # sign flip LH -> RH
        psi_s = interpax.interp1d(s, s_full, psi_s)

        xm = file.variables["ixm_b"][:].filled()
        xn = file.variables["ixn_b"][:].filled()

        B0 = jnp.abs(b_mnc).max()
        mask = jnp.abs(b_mnc) > cutoff * B0

        Bmag = vmec_eval(theta[:, None], zeta[None, :], b_mnc * mask, 0, xm, -xn)
        dBdt = vmec_eval(theta[:, None], zeta[None, :], b_mnc * mask, 0, xm, -xn, dt=1)
        dBdz = vmec_eval(theta[:, None], zeta[None, :], b_mnc * mask, 0, xm, -xn, dz=1)

        sign = jnp.sign(bvco + iota * buco)
        buco *= sign
        bvco *= sign
        sqrtg = (bvco + iota * buco) / Bmag**2
        data = {}
        data["sqrtg"] = sqrtg
        data["Bmag"] = Bmag
        data["dBdt"] = dBdt
        data["dBdz"] = dBdz
        data["B_sub_t"] = buco * jnp.ones((ntheta, nzeta))
        data["B_sub_z"] = bvco * jnp.ones((ntheta, nzeta))
        data["B_sup_t"] = iota / sqrtg
        data["B_sup_z"] = 1 / sqrtg
        data["psi_r"] = psi_s * 2 * jnp.sqrt(s) / a_minor
        data["iota"] = iota
        data["B0"] = B0

        return cls(rho=jnp.sqrt(s), **data, NFP=nfp, deriv_mode=deriv_mode)

    def flux_surface_average(self, f: Float[Array, "ntheta nzeta"]) -> float:
        """Compute flux surface average of f."""
        f = f.reshape((-1, self.ntheta, self.nzeta))
        g = f * self.sqrtg
        return g.mean(axis=(-1, -2)) / self.sqrtg.mean()

    @functools.partial(jnp.vectorize, signature="(m,n)->(m,n)", excluded=[0])
    def bdotgrad(self, f: Float[Array, "ntheta nzeta"]) -> Float[Array, "ntheta nzeta"]:
        """𝐛 ⋅ ∇ f."""
        return (self.B_sup_t * self._dfdt(f) + self.B_sup_z * self._dfdz(f)) / self.Bmag

    @functools.partial(jnp.vectorize, signature="(m,n)->(m,n)", excluded=[0])
    def Bxgradpsidotgrad(
        self, f: Float[Array, "ntheta nzeta"]
    ) -> Float[Array, "ntheta nzeta"]:
        """𝐁 × ∇ ψ ⋅ ∇ f."""
        return (
            self.B_sub_z * self._dfdt(f) - self.B_sub_t * self._dfdz(f)
        ) / self.sqrtg

    @functools.partial(jnp.vectorize, signature="(m,n)->(m,n)", excluded=[0])
    def _dfdt(self, f: Float[Array, "ntheta nzeta"]) -> Float[Array, "ntheta nzeta"]:
        if self.deriv_mode == "fft":
            g = jnp.fft.fft(f, axis=0)
            k = jnp.fft.fftfreq(self.ntheta, 1 / self.ntheta)
            df = jnp.fft.ifft(1j * k[:, None] * g, axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return df.astype(f.dtype)
        else:
            coeffs = {
                "fd2": jnp.array([-1 / 2, 0, 1 / 2]),
                "fd4": jnp.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]),
                "fd6": jnp.array([-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60]),
            }
            d = coeffs[self.deriv_mode] / (2 * np.pi / self.ntheta)
            m = len(d)
            f = jnp.concatenate([f[-m:], f, f[:m]], axis=0)
            convolve = lambda x: jnp.convolve(x, d[::-1], "same")
            df = jax.vmap(convolve)(f.T).T
            return df[m:-m]

    @functools.partial(jnp.vectorize, signature="(m,n)->(m,n)", excluded=[0])
    def _dfdz(self, f: Float[Array, "ntheta nzeta"]) -> Float[Array, "ntheta nzeta"]:
        if self.deriv_mode == "fft":
            g = jnp.fft.fft(f, axis=1)
            k = jnp.fft.fftfreq(self.nzeta, 1 / self.nzeta) * self.NFP
            df = jnp.fft.ifft(1j * k[None, :] * g, axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return df.astype(f.dtype)
        else:
            coeffs = {
                "fd2": jnp.array([-1 / 2, 0, 1 / 2]),
                "fd4": jnp.array([1 / 12, -2 / 3, 0, 2 / 3, -1 / 12]),
                "fd6": jnp.array([-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60]),
            }
            d = coeffs[self.deriv_mode] / (2 * np.pi / self.nzeta / self.NFP)
            m = len(d)
            f = jnp.concatenate([f[:, -m:], f, f[:, :m]], axis=1)
            convolve = lambda x: jnp.convolve(x, d[::-1], "same")
            df = jax.vmap(convolve)(f)
            return df[:, m:-m]

    def resample(self, ntheta: int, nzeta: int):
        """Resample field to a higher resolution."""
        if ntheta == self.ntheta and nzeta == self.nzeta:
            return self

        keys = [
            "B_sup_t",
            "B_sup_z",
            "B_sub_t",
            "B_sub_z",
            "sqrtg",
            "Bmag",
            "dBdt",
            "dBdz",
        ]
        out = {}
        for key in keys:
            out[key] = interpax.fft_interp2d(getattr(self, key), ntheta, nzeta)
        return Field(
            self.rho,
            **out,
            psi_r=self.psi_r,
            B0=self.B0,
            NFP=self.NFP,
            deriv_mode=self.deriv_mode,
        )


# TODO: make this a lineax.LinearOperator
class MonoenergeticDKOperator(eqx.Module):
    """Linear operator representing LHS of mono-energetic drift kinetic equation.

    Parameters
    ----------
    field : Field
        Magnetic field information.
    nl : int
        Number of Legendre modes in pitch angle coordinate.
    Erhat : float
        Radial electric field normalized by velocity
    nuhat :
        Collisionality normalized by velocity.
    """

    field: Field
    nl: int = eqx.field(static=True)
    Erhat: float
    nuhat: float
    shape: tuple = eqx.field(static=True)

    def __init__(self, field, nl, Erhat, nuhat):
        self.field = field
        self.Erhat = Erhat
        self.nuhat = nuhat
        self.nl = nl
        self.shape = (nl * field.ntheta * field.nzeta, nl * field.ntheta * field.nzeta)

    @jit
    def mv(self, x):
        """Matrix vector product."""
        x = x.reshape((self.nl, self.field.ntheta, self.field.nzeta))
        k = jnp.arange(self.nl)
        N = self.field.ntheta * self.field.nzeta
        a = self._Uk(x[1:], k[:-1]).flatten()
        b = self._Dk(x[:], k[:]).flatten()
        c = self._Lk(x[:-1], k[1:]).flatten()
        return b.at[:-N].add(a).at[N:].add(c)

    def _maybe_flatten(self, op, f, k):
        flatten = False
        if f.ndim == 1:
            f = f.reshape((-1, self.field.ntheta, self.field.nzeta))
            flatten = True
        out = op(f, k)
        if flatten:
            out = out.reshape(-1)
        return out

    @jit
    @functools.partial(jnp.vectorize, signature="()->(n,n)", excluded=[0])
    def get_Lkmat(self, k):
        f = np.zeros(self.field.ntheta * self.field.nzeta)

        def Lk(f, k):
            return self._maybe_flatten(self._Lk, f, k)

        return jax.jacfwd(Lk)(f, k)

    @jit
    @functools.partial(jnp.vectorize, signature="()->(n,n)", excluded=[0])
    def get_Dkmat(self, k):
        f = np.zeros(self.field.ntheta * self.field.nzeta)

        def Dk(f, k):
            return self._maybe_flatten(self._Dk, f, k)

        return jax.jacfwd(Dk)(f, k)

    @jit
    @functools.partial(jnp.vectorize, signature="()->(n,n)", excluded=[0])
    def get_Ukmat(self, k):
        f = np.zeros(self.field.ntheta * self.field.nzeta)

        def Uk(f, k):
            return self._maybe_flatten(self._Uk, f, k)

        return jax.jacfwd(Uk)(f, k)

    @functools.partial(jnp.vectorize, signature="(m,n),()->(m,n)", excluded=[0])
    def _Lk(self, f, k):
        return (
            k
            / (2 * k - 1)
            * (
                self.field.bdotgrad(f)
                + (k - 1) / 2 * self.field.bdotgradB * f / self.field.Bmag
            )
        )

    @functools.partial(jnp.vectorize, signature="(m,n),()->(m,n)", excluded=[0])
    def _Dk(self, f, k):

        Df = (
            -self.Erhat
            / self.field.psi_r
            / self.field.B2mag_fsa
            * self.field.Bxgradpsidotgrad(f)
            + k * (k + 1) / 2 * self.nuhat * f
        )
        return jnp.where(k == 0, Df.at[0, 0].set(f[0, 0]), Df)

    @functools.partial(jnp.vectorize, signature="(m,n),()->(m,n)", excluded=[0])
    def _Uk(self, f, k):
        Uf = (
            (k + 1)
            / (2 * k + 3)
            * (
                self.field.bdotgrad(f)
                - (k + 2) / 2 * self.field.bdotgradB * f / self.field.Bmag
            )
        )
        return jnp.where(k == 0, Uf.at[0, 0].set(0.0), Uf)


def vmec_eval(t, z, xc, xs, m, n, dt=0, dz=0):
    """Evaluate a vmec style double-fourier series.

    eg sum_mn xc*cos(m*t-n*z) + xs*sin(m*t-n*z)

    Parameters
    ----------
    t, z : float, jax.Array
        theta, zeta coordinates to evaluate at.
    xc, xs : jax.Array
        Cosine, sine coefficients of double fourier series.
    m, n : jax.Array
        Poloidal and toroidal mode numbers.

    Returns
    -------
    x : float, jax.Array
        Evaluated quantity at t, z.
    """
    xc, xs, m, n, dt, dz = jnp.atleast_1d(xc, xs, m, n, dt, dz)
    xc, xs, m, n, dt, dz = jnp.broadcast_arrays(xc, xs, m, n, dt, dz)
    return _vmec_eval(t, z, xc, xs, m, n, dt, dz)


@functools.partial(jnp.vectorize, signature="(),(),(n),(n),(n),(n),(n),(n)->()")
def _vmec_eval(t, z, xc, xs, m, n, dt, dz):
    arg = m * t - n * z
    arg += dt * jnp.pi / 2
    arg -= dz * jnp.pi / 2
    xc *= m**dt
    xc *= n**dz
    xs *= m**dt
    xs *= n**dz
    c = (xc * jnp.cos(arg)).sum()
    s = (xs * jnp.sin(arg)).sum()
    return c + s