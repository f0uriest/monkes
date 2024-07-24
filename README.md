# monkes
Monoenergetic Kinetic Equation Solver

This is a python/JAX port of the [original fortran implementation](https://github.com/JavierEscoto/MONKES).
It solves the drift kinetic equation (DKE) using the monoenergetic approximation,
similar to DKES, but uses JAX so it runs on GPUs and is differentiable.


Basic usage:

```python
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)

import monkes

ne = 5e19
te = 1000
ni = 5e19
ti = 1000

electrons = monkes.GlobalMaxwellian(monkes.Electron, lambda x: te*(1-x**2), lambda x: ne*(1-x**4))
ions = monkes.GlobalMaxwellian(monkes.Hydrogen, lambda x: ti*(1-x**2), lambda x: ni*(1-x**4))

species = [electrons, ions]

import desc
nt = 19
nz = 31
eq = desc.examples.get("HELIOTRON")
field = monkes.Field.from_desc(eq, 0.5, nt, nz)

Dij, f, s = monkes.monoenergetic_dke_solve(field, species, Er=1.0, v=1e5, nl=80)
```

This computes the monoenergetic transport coefficients `Dij`, the perturbed distribution
function `f` and the sources `s`


To Do:
- benchmark/validate against MONKES/DKES
    - Probable issues:
        - normalizations / units
        - consistent definitions of radial coordinate (rho vs s vs psi)
- finish implementing option for full 4D equation (eg SFINCS)
- Add utilites for performing scans over collisionality, Er
- fix up interface to allow easier use of `lineax` solvers
