import jax
import jax.numpy as jnp
from jax import jit


@jit
def block_tridiagonal_factor(diagonal, lower_diagonal, upper_diagonal, reverse=False):
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
    diagonal, lower_diagonal, upper_diagonal, vector, kmax
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

    Returns
    -------
    x : jax.Array, shape(K*N)
        Solution vector.
    """

    def factor_scan(carry, _):
        k, Deltainv_kp1 = carry
        Lkp1 = lower_diagonal(k + 1)
        Uk = upper_diagonal(k)

        DeltainvLkp1 = jax.scipy.linalg.lu_solve(Deltainv_kp1, Lkp1)
        Delta_k = diagonal(k) - Uk @ DeltainvLkp1
        Deltainv_k = jax.scipy.linalg.lu_factor(Delta_k)
        return (k - 1, Deltainv_k), (Deltainv_k,)

    Deltainv_kmax = jax.scipy.linalg.lu_factor(diagonal(kmax - 1))
    init_thomas = (kmax - 2, Deltainv_kmax)
    (k, _), (Deltainv,) = jax.lax.scan(
        factor_scan, init_thomas, None, length=kmax - 1, reverse=True
    )
    Deltainv = (
        jnp.concatenate([Deltainv[0], Deltainv_kmax[0][None]]),
        jnp.concatenate([Deltainv[1], Deltainv_kmax[1][None]]),
    )

    block_size = Deltainv[0].shape[1]
    size = Deltainv[0].shape[0]

    s = jnp.asarray(vector).reshape(size, block_size)

    def forwardsub(carry, Deltinv):
        k, sigma_kp1 = carry
        sk = s[k]
        Uk = upper_diagonal(k)
        Deltinv_sigma = jax.scipy.linalg.lu_solve(Deltinv, sigma_kp1)
        sigma_k = sk - Uk @ Deltinv_sigma
        return (k - 1, sigma_k), (sigma_k,)

    sigma_kp1 = s[-1]
    init_forward = (size - 2, sigma_kp1)
    (k, _), (sigma,) = jax.lax.scan(
        forwardsub, init_forward, (Deltainv[0][1:], Deltainv[1][1:]), reverse=True
    )
    sigma = jnp.concatenate([sigma, sigma_kp1[None]])

    def backsub(carry, Deltinv_sigma):
        k, f_km1 = carry
        Deltainvk, sigmak = Deltinv_sigma
        L = lower_diagonal(k)
        fk = jax.scipy.linalg.lu_solve(Deltainvk, (sigmak - L @ f_km1))
        return (k + 1, fk), fk

    init_backsub = (0, jnp.zeros(block_size))
    (k, _), f = jax.lax.scan(backsub, init_backsub, (Deltainv, sigma))
    return f.flatten()
