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
