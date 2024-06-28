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
    B_sup_p : jax.Array, shape(ntheta, nzeta)
        B^psi, contravariant radial component of field.
    B_sup_t : jax.Array, shape(ntheta, nzeta)
        B^theta, contravariant poloidal component of field.
    B_sup_z : jax.Array, shape(ntheta, nzeta)
        B^zeta, contravariant toroidal component of field.
    B_sub_p : jax.Array, shape(ntheta, nzeta)
        B_psi, covariant radial component of field.
    B_sub_t : jax.Array, shape(ntheta, nzeta)
        B_theta, covariant poloidal component of field.
    B_sub_z : jax.Array, shape(ntheta, nzeta)
        B_zeta, covariant toroidal component of field.
    sqrtg : jax.Array, shape(ntheta, nzeta)
        Coordinate jacobian determinant.
    NFP : int
        Number of field periods.
    """

    # note: assumes (psi, theta, zeta) coordinates, not (rho, theta, zeta)
    rho: float
    B_sup_p: Float[Array, "ntheta nzeta"]
    B_sup_t: Float[Array, "ntheta nzeta"]
    B_sup_z: Float[Array, "ntheta nzeta"]
    B_sub_p: Float[Array, "ntheta nzeta"]
    B_sub_t: Float[Array, "ntheta nzeta"]
    B_sub_z: Float[Array, "ntheta nzeta"]
    sqrtg: Float[Array, "ntheta nzeta"]
    Bmag: Float[Array, "ntheta nzeta"]
    bdotgradB: Float[Array, "ntheta nzeta"]
    Bmag_fsa: float
    ntheta: int = eqx.field(static=True)
    nzeta: int = eqx.field(static=True)
    NFP: int = eqx.field(static=True)

    def __init__(
        self, rho, B_sup_p, B_sup_t, B_sup_z, B_sub_p, B_sub_t, B_sub_z, sqrtg, NFP=1
    ):
        self.rho = rho
        self.NFP = NFP
        self.ntheta = sqrtg.shape[0]
        self.nzeta = sqrtg.shape[1]
        assert (self.ntheta % 2 == 1) and (
            self.nzeta % 2 == 1
        ), "ntheta and nzeta must be odd"
        self.B_sup_p = B_sup_p
        self.B_sup_t = B_sup_t
        self.B_sup_z = B_sup_z
        self.B_sub_p = B_sub_p
        self.B_sub_t = B_sub_t
        self.B_sub_z = B_sub_z
        self.sqrtg = sqrtg
        self.Bmag = B_sup_p * B_sub_p + B_sup_t * B_sub_t + B_sup_z * B_sub_z
        self.bdotgradB = (
            B_sup_t * self._dfdt(self.Bmag) + B_sup_z * self._dfdz(self.Bmag)
        ) / self.Bmag
        self.Bmag_fsa = self.flux_surface_average(self.Bmag)

    @classmethod
    def from_desc(cls, eq, rho: int, ntheta: float, nzeta: float):
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
        """
        assert (ntheta % 2 == 1) and (nzeta % 2 == 1), "ntheta and nzeta must be odd"

        from desc.grid import LinearGrid

        grid = LinearGrid(rho=rho, theta=ntheta, zeta=nzeta, endpoint=False, NFP=eq.NFP)
        keys = ["B^rho", "B^theta", "B^zeta", "B_rho", "B_theta", "B_zeta", "sqrt(g)"]
        desc_data = eq.compute(keys, grid=grid)

        data = {
            "B_sup_p": desc_data["B^rho"] * desc_data["psi_r"],
            "B_sup_t": desc_data["B^theta"],
            "B_sup_z": desc_data["B^zeta"],
            "B_sub_p": desc_data["B_rho"] / desc_data["psi_r"],
            "B_sub_t": desc_data["B_theta"],
            "B_sub_z": desc_data["B_zeta"],
            "sqrtg": desc_data["sqrt(g)"] / desc_data["psi_r"],
        }

        data = {
            key: val.reshape((grid.num_theta, grid.num_zeta), order="F")
            for key, val in data.items()
        }
        return cls(rho=rho, **data, NFP=eq.NFP)

    @classmethod
    def from_vmec(cls, wout, s: float, ntheta: int, nzeta: int):
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
        raise NotImplementedError

    @classmethod
    def from_booz_xform(cls, booz, s: float, ntheta: int, nzeta: int):
        """Construct Field from BOOZ_XFORM file.

        Parameters
        ----------
        booz : path-like
            Path to booz_xform wout file.
        s : float
            Flux surface label.
        ntheta, nzeta : int
            Number of points on a surface in poloidal and toroidal directions.
        """
        raise NotImplementedError

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
            self.B_sub_z * self._dfdt(f) + self.B_sub_t * self._dfdz(f)
        ) / self.sqrtg

    @functools.partial(jnp.vectorize, signature="(m,n)->(m,n)", excluded=[0])
    def _dfdt(self, f: Float[Array, "ntheta nzeta"]) -> Float[Array, "ntheta nzeta"]:
        g = jnp.fft.fft(f, axis=0)
        k = jnp.fft.fftfreq(self.ntheta, 1 / self.ntheta)
        df = jnp.fft.ifft(1j * k[:, None] * g, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return df.astype(f.dtype)

    @functools.partial(jnp.vectorize, signature="(m,n)->(m,n)", excluded=[0])
    def _dfdz(self, f: Float[Array, "ntheta nzeta"]) -> Float[Array, "ntheta nzeta"]:
        g = jnp.fft.fft(f, axis=1)
        k = jnp.fft.fftfreq(self.nzeta, 1 / self.nzeta) * self.NFP
        df = jnp.fft.ifft(1j * k[None, :] * g, axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return df.astype(f.dtype)

    def resample(self, ntheta: int, nzeta: int):
        """Resample field to a higher resolution."""
        if ntheta == self.ntheta and nzeta == self.nzeta:
            return self

        keys = [
            "B_sup_p",
            "B_sup_t",
            "B_sup_z",
            "B_sub_p",
            "B_sub_t",
            "B_sub_z",
            "sqrtg",
            "Bmag",
            "bdotgradB",
        ]
        out = {}
        for key in keys:
            out[key] = interpax.fft_interp2d(getattr(self, key), ntheta, nzeta)
        return Field(self.rho, **out)


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
    nl: int
    Erhat: float
    nuhat: float
    D: jax.Array
    L: jax.Array
    U: jax.Array
    shape: tuple

    def __init__(self, field, nl, Erhat, nuhat):
        self.field = field
        self.Erhat = Erhat
        self.nuhat = nuhat
        self.nl = nl
        k = np.arange(nl)
        self.L = self._get_Lkmat(k[1:])
        self.D = self._get_Dkmat(k)
        self.U = self._get_Ukmat(k[:-1])
        self.shape = (nl * field.ntheta * field.nzeta, nl * field.ntheta * field.nzeta)

    def mv(self, x):
        """Matrix vector product."""
        size, N, M = jnp.shape(self.D)
        v = x.reshape(size, N)
        a = jnp.einsum("ijk,ik -> ij", self.U, v[1:, :]).flatten()
        b = jnp.einsum("ijk,ik -> ij", self.D, v[:, :]).flatten()
        c = jnp.einsum("ijk,ik -> ij", self.L, v[:-1, :]).flatten()
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
    def _get_Lkmat(self, k):
        f = np.zeros(self.field.ntheta * self.field.nzeta)

        def Lk(f, k):
            return self._maybe_flatten(self._Lk, f, k)

        return jax.jacfwd(Lk)(f, k)

    @jit
    @functools.partial(jnp.vectorize, signature="()->(n,n)", excluded=[0])
    def _get_Dkmat(self, k):
        f = np.zeros(self.field.ntheta * self.field.nzeta)

        def Dk(f, k):
            return self._maybe_flatten(self._Dk, f, k)

        D = jax.jacfwd(Dk)(f, k)
        return jnp.where(k == 0, D.at[0, :].set(0.0).at[0, 0].set(1.0), D)

    @jit
    @functools.partial(jnp.vectorize, signature="()->(n,n)", excluded=[0])
    def _get_Ukmat(self, k):
        f = np.zeros(self.field.ntheta * self.field.nzeta)

        def Uk(f, k):
            return self._maybe_flatten(self._Uk, f, k)

        U = jax.jacfwd(Uk)(f, k)
        return jnp.where(k == 0, U.at[0, :].set(0.0), U)

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
        return (
            -self.Erhat / self.field.Bmag_fsa * self.field.Bxgradpsidotgrad(f)
            + k * (k + 1) / 2 * self.nuhat * f
        )

    @functools.partial(jnp.vectorize, signature="(m,n),()->(m,n)", excluded=[0])
    def _Uk(self, f, k):
        return (
            (k + 1)
            / (2 * k + 3)
            * (
                self.field.bdotgrad(f)
                + (k + 2) / 2 * self.field.bdotgradB * f / self.field.Bmag
            )
        )
