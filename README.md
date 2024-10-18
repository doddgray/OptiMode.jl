# OptiMode.jl
This code is a prototype differentiable eigensolver for the electromagnetic Helmholtz equation.

The mathematical model implemented in this code is a re-implementation of the plane-wave, transverse-polarization basis electromagnetic mode solver [MIT Photonic Bands (MPB)](https://mpb.readthedocs.io/en/latest/) [[1]](#1) and is also described in our paper [[2]](#2).
This package extends the functionality of MPB by implementing a "pull-back" (vector-Jacobian product) function to back-propagate gradients with respect electromagnetic mode fields and spatial propagation constants to the parameters determining the modes, namely the dielectric tensor elements at each spatial grid point and the (temporal) frequency of the electromagnetic modes.
The "pull-back" function works by iteratively solving the adjoint equations for the electromagnetic Helmholtz eigen-problem.
As demonstrated in our paper, these gradients can be further back-propagated to parameters defining an optical waveguide geometry and used to optimize a waveguide for some desired modal properties.

Currently the code is undocumented.
We took a long hiatus from developing this code.
In the near future we plan to update and refactor it to make it faster and add functionality.
Nonetheless, this old code still works (as of Julia version 1.10).
We will try to add some basic use examples here in the `README` and as scripts very soon.
In the meantime if you would like to use the code, please post an issue because we would like to help you install and run it.

If you find this solver useful in your own research please consider citing our paper [[2]](#2) and the original MPB paper [[1]](#1).
If you find this solver broken or buggy please post an issue so that we can try to fix and improve it.
Good luck and happy mode solving.

## References
<a id="1">[1]</a> 
S. G. Johnson and J. D. Joannopoulos, "Block-iterative frequency-domain methods for Maxwell’s equations in a planewave basis," [Optics Express 8, 173-190 (2001)](https://doi.org/10.1364/OE.8.000173)

<a id="2">[2]</a> 
D. Gray, G. N. West, and R. J. Ram, "Inverse design for waveguide dispersion with a differentiable mode solver," [Optics Express 32, 30541-30554 (2024)](https://doi.org/10.1364/OE.530479)