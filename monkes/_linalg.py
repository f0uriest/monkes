from functools import partial

import jax
import jax.numpy as jnp
from jax import jit


@partial(jit, static_argnames=("debug",))
def block_tridiagonal_factor(
    diagonal, lower_diagonal, upper_diagonal, reverse=False, debug=False
):
    """Factor a block tridiagonal matrix for later use.

    Parameters
    ----------
    diagonal : jax.Array, shape(n,m,m)
        Main diagonal block
    lower_diagonal, upper_diagonal : jax.Array, shape(n-1,m,m)
        Lower and upper diagonal blocks
    reverse: bool
        If True, start at D[-1] and go backwards. This may be necessary in cases
        where D[0] is rank deficient but the Schur compliment of D[0] is still
        invertible.

    Returns
    -------
    Clu : tuple
        LU factored blocks and auxiliary information needed for
        block_tridiagonal_solve
    """
    diagonal, lower_diagonal, upper_diagonal = map(
        jnp.asarray, (diagonal, lower_diagonal, upper_diagonal)
    )
    block_size = diagonal.shape[1]
    size = diagonal.shape[0]

    def reverse_true():
        # lower and upper are swapped
        return (
            jnp.flipud(diagonal),
            jnp.flipud(upper_diagonal),
            jnp.flipud(lower_diagonal),
        )

    def reverse_false():
        return diagonal, lower_diagonal, upper_diagonal

    diagonal, lower_diagonal, upper_diagonal = jax.lax.cond(
        reverse, reverse_true, reverse_false
    )

    def factor_scan(carry, D):
        step, C = carry
        L_index = jnp.where(step > 0, step - 1, 0)
        U_index = jnp.where(step < size, step, 0)
        L, U = lower_diagonal[L_index, :, :], upper_diagonal[U_index, :, :]

        denom = D - jnp.matmul(L, C)
        if debug:
            jax.debug.print("cond={x}", x=jnp.linalg.cond(denom))
        lu = jax.scipy.linalg.lu_factor(denom)
        new_C = jax.scipy.linalg.lu_solve(lu, U)
        return (step + 1, new_C), (new_C, lu)

    init_thomas = (0, jnp.zeros((block_size, block_size)))
    _, (C, lu) = jax.lax.scan(factor_scan, init_thomas, diagonal)
    return C, lu, lower_diagonal, reverse


@jit
def block_tridiagonal_solve(Clu, vector):
    """Factor a block tridiagonal matrix for later use.

    Parameters
    ----------
    Clu : tuple
        LU factored blocks and auxiliary information, output from
        block_tridiagonal_factor
    vector : jax.Array, shape(n*m)
        RHS vector to solve against.

    Returns
    -------
    x : jax.Array, shape(n*m)
        Solution vector.
    """
    C, lu, lower_diagonal, reverse = Clu
    block_size = C.shape[1]
    size = C.shape[0]

    b = vector.reshape(size, block_size)
    b = jnp.where(reverse, jnp.flipud(b), b)

    def forwardsub(carry, lub):
        step, E = carry
        lu, b = lub

        L_index = jnp.where(step > 0, step - 1, 0)
        L = lower_diagonal[L_index, :, :]

        new_E = jax.scipy.linalg.lu_solve(lu, b - jnp.matmul(L, E))
        return (step + 1, new_E), (new_E,)

    def backsub(x, CE):
        Ck, Ek = CE
        x = Ek - jnp.dot(Ck, x)
        return x, x

    init_forward = (0, jnp.zeros(block_size))
    init_backsub = jnp.zeros(block_size)
    _, (E,) = jax.lax.scan(forwardsub, init_forward, (lu, b))
    _, solution = jax.lax.scan(backsub, init_backsub, (C, E), reverse=True)
    solution = jnp.where(reverse, jnp.flipud(solution), solution)
    return solution.flatten()


def block_tridiag_mv(D, L, U, x):
    """Matrix vector product for block tridiagonal matrix."""
    size, N, M = jnp.shape(D)
    v = x.reshape(size, N)
    a = jnp.einsum("ijk,ik -> ij", U, v[1:, :]).flatten()
    b = jnp.einsum("ijk,ik -> ij", D, v[:, :]).flatten()
    c = jnp.einsum("ijk,ik -> ij", L, v[:-1, :]).flatten()
    return b.at[:-N].add(a).at[N:].add(c)


def block_tridiagonal_solve_lazy(
    diagonal, lower_diagonal, upper_diagonal, vector, Lmax, debug=False
):
    """Solve a block tridiagonal system using limited memory.

    Parameters
    ----------
    diagonal : callable
        Function to calculate main diagonal block for given k, signature ()->(N,N)
    lower_diagonal, callable
        Functions to calculate lower and upper diagonal blocks for given k,
        signature ()->(N,N)
    vector : jax.Array, shape(K*N)
        RHS vector to solve against.
    Lmax : int
        Maximum value for desired output. For monoenergetic coefficients, only needs
        Lmax=2. For full distribution function, needs Lmax=kmax.

    Returns
    -------
    x : jax.Array, shape(K*N)
        Solution vector.
    """
    block_size = jax.eval_shape(diagonal, 0).shape[0]
    kmax = len(vector) // block_size

    Dk = diagonal(kmax - 1)
    Deltainv = jnp.zeros((Lmax + 1, block_size, block_size))
    pivots = jnp.zeros((Lmax + 1, block_size), dtype=jnp.int32)
    sigma = jnp.zeros((Lmax + 1, block_size))
    s = jnp.asarray(vector).reshape(kmax, block_size)

    def factor_body(i, carry):
        k = kmax - i
        Deltainv, pivots, sigma, Deltainv_kp1, pivots_kp1, sigma_kp1 = carry
        Lkp1 = lower_diagonal(k + 1)
        Uk = upper_diagonal(k)

        DeltainvLkp1 = jax.scipy.linalg.lu_solve((Deltainv_kp1, pivots_kp1), Lkp1)
        Deltainvskp1 = jax.scipy.linalg.lu_solve((Deltainv_kp1, pivots_kp1), sigma_kp1)
        Delta_k = diagonal(k) - Uk @ DeltainvLkp1
        sigma_k = s[k] - Uk @ Deltainvskp1
        if debug:
            jax.debug.print("cond={x}", x=jnp.linalg.cond(Delta_k))

        Deltainv_k, pivots_k = jax.scipy.linalg.lu_factor(Delta_k)

        def kltLmax(Deltainv, pivots, sigma):
            Deltainv = Deltainv.at[k].set(Deltainv_k)
            pivots = pivots.at[k].set(pivots_k)
            sigma = sigma.at[k].set(sigma_k)
            return Deltainv, pivots, sigma

        def kgtLmax(Deltainv, pivots, sigma):
            return Deltainv, pivots, sigma

        Deltainv, pivots, sigma = jax.lax.cond(
            k <= Lmax, kltLmax, kgtLmax, Deltainv, pivots, sigma
        )

        return (Deltainv, pivots, sigma, Deltainv_k, pivots_k, sigma_k)

    if debug:
        jax.debug.print("cond={x}", x=jnp.linalg.cond(Dk))

    Deltainv_kp1, pivots_kp1 = jax.scipy.linalg.lu_factor(Dk)
    sigma_kp1 = s[kmax]
    init_thomas = (Deltainv, pivots, sigma, Deltainv_kp1, pivots_kp1, sigma_kp1)

    (Deltainv, pivots, sigma, _, _, _) = jax.lax.fori_loop(
        2, kmax + 1, factor_body, init_thomas
    )

    def backsub(carry, Deltinv_sigma):
        k, f_km1 = carry
        Deltainvk, pivots, sigmak = Deltinv_sigma
        L = lower_diagonal(k)
        fk = jax.scipy.linalg.lu_solve((Deltainvk, pivots), (sigmak - L @ f_km1))
        return (k + 1, fk), fk

    init_backsub = (0, jnp.zeros(block_size))
    (k, _), f = jax.lax.scan(backsub, init_backsub, (Deltainv, pivots, sigma))
    ff = jnp.zeros((kmax, block_size)).at[: Lmax + 1].set(f)
    return ff.flatten()
