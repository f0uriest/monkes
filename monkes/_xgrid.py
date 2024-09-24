import functools
import warnings

import equinox as eqx
import quadax as qdx
import jax
import numpy as np
import jax.numpy as jnp
from jax import jit
from jaxtyping import Array, Float


class xGrid(eqx.Module):
    """Energy x=v/vth grid

    Contains the Nx nodes (abscissae) and respective weights (xWeights) for the energy integration

    Parameters
    ----------
    Nx : float
        Number of grid/nodes points
    """

    # note: assumes (psi, theta, zeta) coordinates, not (rho, theta, zeta)
    Nx: float
    abscissae: Float[Array, "Nx "]
    xWeights: Float[Array, "Nx "]
    pointAtX0: bool

    def __init__(
        self, Nx, pointAtX0, abscissae, xWeights
    ):
        self.Nx = Nx
        self.pointAtX0 = pointAtX0
        self.abscissae = abscissae
        self.xWeights = xWeights


    @classmethod
    def sfincsGrid(cls, Nx: float, pointAtX0: bool) :
        """Constructs sfincs-like energy grid for obtaining exact abscissae and weights 
        in which to perform monoenergetic simulatiosn.

        Parameters
        ----------
        Nx : Number of grid points
        pintAtX0 : Will this point be considered, usually false in sfincs, and to be implemented here
        """
        #assert (ntheta % 2 == 1) and (nzeta % 2 == 1), "ntheta and nzeta must be odd"


        # This is one of 2 main subroutines of the module.
        #
        # Inputs:
        #  N = number of grid points to generate.
        #  includePointAtX0 = Should a point be included at x=0?
        #
        # Outputs:
        #   abscissae = grid points
        #   weights = Gaussian integration weights
    
        abscissae=jnp.zeros(Nx)
        xWeights=jnp.zeros(Nx)
        a=jnp.zeros(Nx)
        b=jnp.zeros(Nx+1)
        a_copy=jnp.zeros(Nx)
        sqrtb=jnp.zeros(Nx+1)
        c=jnp.zeros(Nx)
        d=jnp.zeros(Nx)
        eigenvectors=jnp.zeros((Nx,Nx))
        eigenvalues=jnp.zeros(Nx)
        epsabs = 0.0
        epsrel = 1e-13
        oldc=1.0

        #X0 = 0.0d+0  ! Special point to include among the abscissae, if requested.
        #lastPolynomialAtX0 = 0.0d+0
        #penultimatePolynomialAtX0 = 0.0d+0

        interval= jnp.array([0,jnp.inf])
        #@functools.partial(jnp.vectorize, signature="(m)->(m)", excluded=[0])
        def weight_Function(x: float):
            """𝐛 ⋅ ∇ f."""
            xGrid_k=0.  #See this later
            return jnp.exp(-x*x)*(x ** xGrid_k)



        #functools.partial(jnp.vectorize, signature="(m)->(m)", excluded=[0])
        def evaluatePolynomial(x: float):
            """𝐁 × ∇ ψ ⋅ ∇ f."""

            y = 0.0

            print('inside',j)

            if j == 0:
                evaluatePolynomial = 1.0
            else:
                pjMinus1 = 0.0
                pj = 1.0
                for ii in range(j):
                    print('poly',ii) 
                    print('a[ii]',a)
                    print('b[ii]', b)
                    y = (x-a[ii]) * pj - b[ii] * pjMinus1
                    pjMinus1 = pj
                    pj = y
                #End for loop
                evaluatePolynomial = y

            return evaluatePolynomial

        #functools.partial(jnp.vectorize, signature="(m)->(m)", excluded=[0])
        def integrandWithoutX(x: float): 
            p = evaluatePolynomial(x)
            integrandWithoutX = p*weight_Function(x)*p

            return integrandWithoutX


        #functools.partial(jnp.vectorize, signature="(m)->(m)", excluded=[0])
        def integrandWithX( x: float):
            p = evaluatePolynomial(x)
            integrandWithX = x*p*weight_Function(x)*p

            return integrandWithX


        #@functools.partial(jnp.vectorize, signature="(m)->(m)", excluded=[0])
        def integrandWithPower(x: float) :
            p = evaluatePolynomial(x)
            # Note that x**xGrid_k should not be included in the next line!
            integrationPower=0.0
            integrandWithPower = (x**integrationPower)*p*jnp.exp(-x*x)

            return integrandWithPower


        def tridiag(a, b, c, k1=-1, k2=0, k3=1):
            return jnp.diag(a, k1) + jnp.diag(b, k2) + jnp.diag(c, k3)

        #This part calculates integrals, with j being an index for the polynomials used in the integrands
        #Instead of quadpack of fortran, using quadax (jax/Rory)
        for j in range(len(a)):
            print(j)
            #Integrate integrandwithoutX for polunomial with index j, result is in c vector
            c_value,info1=qdx.quadgk(integrandWithoutX,interval,epsabs=epsabs,epsrel=epsrel)
            d_value,info2=qdx.quadgk(integrandWithX,interval,epsabs=epsabs,epsrel=epsrel)


            print('c_value', c_value)
            print('d_value', d_value)

            c=c.at[j].set(c_value)
            d=d.at[j].set(d_value)            

            #The we calculate b and a and updat oldC
            b=b.at[j].set(c_value/oldc)
            a=a.at[j].set(d_value/c_value)
            oldc = c_value
 
        a_copy = a
        sqrtb = jnp.sqrt(b)

        #print(a_copy)
        #print(b)
        #print(sqrtb)
   
        M_tri= tridiag(sqrtb[1:-1],a_copy,sqrtb[1:-1]) 
        eigenvalues, eigenvectors=jax.scipy.linalg.eigh(M_tri) 
        #print('a_copy',a_copy)
        #print('Matrix',M_tri)
        #print('lambdas',eigenvalues)
        #print('eigen_vec',eigenvectors)


        for i in range(Nx):
            #abscissae=abscissae.at[i].set(a_copy[Nx-1-i])
            abscissae=abscissae.at[i].set(eigenvalues[i])
            #xWeights=xWeights.at[i].set(c[0]* eigenvectors[0, Nx-1-i]*eigenvectors[0,Nx-1-i])
            xWeights=xWeights.at[i].set(c[0]* eigenvectors[0, i]*eigenvectors[0,i])
        print('abscissae',abscissae)
        print('xWeights',xWeights)

        return cls(Nx=Nx, pointAtX0=False , abscissae=abscissae, xWeights=xWeights)






"""@functools.partial(jnp.vectorize, signature="(m)->(m)", excluded=[0])
def weight_Function(self, x: Float[Array, "Nx"]) -> Float[Array, "Nx"]:

    xGrid_k=0.  #See this later
    return jnp.exp(-x*x)*(x ** xGrid_k)



@functools.partial(jnp.vectorize, signature="(m)->(m)", excluded=[0])
def evaluatePolynomial(
    self, x: Float[Array, "Nx"]
    ) -> Float[Array, "Nx"]:


    pjMinus1=0.0
    pj=0.0
    y = 0.0

    if j == 1:
        evaluatePolynomial = 1.0
    else:
        pjMinus1 = 0.0
        pj = 1.0
        for ii in range(j-1):
            y = (x-a[ii]) * pj - b[ii] * pjMinus1
            pjMinus1 = pj
            pj = y
        #End for loop
        evaluatePolynomial = y

    return evaluatePolynomial

@functools.partial(jnp.vectorize, signature="(m)->(m)", excluded=[0])
def integrandWithoutX(x: Float[Array, "Nx"]) -> Float[Array, "Nx"]:
    p = evaluatePolynomial(x)
    integrandWithoutX = p*weight(x)*p

    return integrandWithoutX


@functools.partial(jnp.vectorize, signature="(m)->(m)", excluded=[0])
def integrandWithX( x: Float[Array, "Nx"]) -> Float[Array, "Nx"]:
    p = evaluatePolynomial(x)
    integrandWithX = x*p*weight(x)*p

    return integrandWithX


@functools.partial(jnp.vectorize, signature="(m)->(m)", excluded=[0])
def integrandWithPower(x: Float[Array, "Nx"]) -> Float[Array, "Nx"]:
    p = evaluatePolynomial(x)
    # Note that x**xGrid_k should not be included in the next line!
    integrandWithPower = (x**integrationPower)*p*jnp.exp(-x*x)

    return integrandWithPower"""

















