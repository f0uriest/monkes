import functools

import jax
import jax.numpy as jnp
from jax import jit

from ._field import MonoenergeticDKOperator
from ._linalg import (
    block_tridiagonal_factor,
    block_tridiagonal_solve,
    block_tridiagonal_solve_lazy,
)
from ._species import collisionality


@jit
@functools.partial(jnp.vectorize, signature="()->(m,n)", excluded=[0])
def _source1(field, k):
    g = field.BxgradpsidotgradB / (2 * field.Bmag**3)
    prefactor = jnp.where(k == 0, 4 / 3, jnp.where(k == 2, 2 / 3, 0))
    out = prefactor * g
    return jnp.where(k == 0, out.at[0, 0].set(0.0), out)


@jit
@functools.partial(jnp.vectorize, signature="()->(m,n)", excluded=[0])
def _source2(field, k):
    return _source1(field, k)


@jit
@functools.partial(jnp.vectorize, signature="()->(m,n)", excluded=[0])
def _source3(field, k):
    g = field.Bmag / field.B0
    prefactor = jnp.where(k == 1, 1, 0)
    out = prefactor * g
    return jnp.where(k == 0, out.at[0, 0].set(0.0), out)


def sources(field, nl):
    """RHS source terms for monoenergetic DKE."""
    k = jnp.arange(nl)
    return jnp.array(
        [
            _source1(field, k).flatten(),
            _source2(field, k).flatten(),
            _source3(field, k).flatten(),
        ]
    )


@jit
def compute_monoenergetic_coefficients(f, s, field):
    """Compute D_ij coefficients from solution for distribution function f."""
    f = f.reshape((3, -1, field.ntheta, field.nzeta))
    s = s.reshape((3, -1, field.ntheta, field.nzeta))
    D11 = 2 * field.flux_surface_average(
        s[0, 0] * f[0, 0]
    ) + 2 / 5 * field.flux_surface_average(s[0, 2] * f[0, 2])
    D12 = D11
    D13 = 2 * field.flux_surface_average(
        s[0, 0] * f[2, 0]
    ) + 2 / 5 * field.flux_surface_average(s[0, 2] * f[2, 2])
    D21 = D11
    D22 = D11
    D23 = D13
    D31 = 2 / 3 * field.flux_surface_average(field.Bmag / field.B0 * f[0, 1])
    D32 = D31
    D33 = 2 / 3 * field.flux_surface_average(field.Bmag / field.B0 * f[2, 1])
    Dij = jnp.array([[D11, D12, D13], [D21, D22, D23], [D31, D32, D33]])

    return Dij.squeeze()


@functools.partial(jax.jit, static_argnames=("nl", "lazy"))
def monoenergetic_dke_solve_internal(field, nl, Erhat, nuhat, lazy=False):
    """Solve MDKE with normalized inputs."""
    operator = MonoenergeticDKOperator(field, nl, Erhat, nuhat)
    s = sources(field, nl)

    if lazy:

        def _solve(vec):
            return block_tridiagonal_solve_lazy(
                operator.get_Dkmat, operator.get_Lkmat, operator.get_Ukmat, vec, nl
            )

    else:
        k = jnp.arange(nl)
        Clu = block_tridiagonal_factor(
            operator.get_Dkmat(k),
            operator.get_Lkmat(k[1:]),
            operator.get_Ukmat(k[:-1]),
            reverse=True,
        )

        def _solve(vec):
            return block_tridiagonal_solve(Clu, vec)

    f = jax.vmap(_solve)(s)

    Dij = compute_monoenergetic_coefficients(f, s, field)
    return Dij, f, s


def monoenergetic_dke_solve(field, global_maxwellians, Er, v, nl, nt=None, nz=None):
    """Solve the monoenergetic drift kinetic equation.

    Parameters
    ----------
    field : monkes.Field
        Magnetic field information.
    global_maxwellians : iterable of monkes.GlobalMaxwellian
        Maxwellian distributions of species involved.
    Er : float
        Radial electric field.
    v : float
        Collision speed.
    nl : int
        Number of Legendre modes in pitch angle coordinates.
    nt, nz : int
        Number of points on flux surface in theta, zeta. Defaults to values from field.

    Returns
    -------
    Dij : jax.Array, shape(3,3)
        Monoenergetic coefficients.
    f : jax.Array, shape(3, nt*nz*nl)
        Perturbed distribution function.
    s : jax.Array, shape(3, nt*nz*nl)
        Source terms (RHS of DKE)

    """
    if nt is None:
        nt = field.ntheta
    if nz is None:
        nz = field.nzeta
    field = field.resample(nt, nz)
    local_maxwellians = [m.localize(field.rho) for m in global_maxwellians]
    nu = collisionality(local_maxwellians[0], v, *local_maxwellians[1:])

    return monoenergetic_dke_solve_internal(field, nl, Er / v, nu / v)
