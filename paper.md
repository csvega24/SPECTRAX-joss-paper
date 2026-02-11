---
title: 'SPECTRAX: A JAX-based Spectral Vlasov-Maxwell Solver'
tags:
  - plasma physics
  - kinetic theory
  - Vlasov-Maxwell
  - spectral methods
  - JAX
authors:
  - name: Cristian Vega
    affiliation: 1
  - name: Rogerio Jorge
    affiliation: 1
  - name: Vladimir Zhdankin
    affiliation: 1
  - name: Benjamin Herfray
    affiliation: 1
affiliations:
  - name: University of Wisconsin–Madison
    index: 1
date: 2026-02-11
bibliography: paper.bib
---

# Summary

Resolving the multiple scales of weakly collisional plasmas poses a computational challenge. One possible approach is to perform a spectral decomposition of the one-particle probability density function (PDF) and the electric and magnetic fields, and keep only the modes needed to model the physics of interest. SPECTRAX is a high-performance, open-source Vlasov-Maxwell solver that models the dynamics of collisionless plasmas by time-evolving the Hermite-Fourier moments of the one-particle PDF and the fields. It is written in pure Python and leverages the JAX library for just-in-time (JIT) compilation, automatic parallelization, and execution on hardware accelerators. Time integration is performed using the Diffrax library, which provides high-order, adaptive, and efficient time-stepping of the resulting system of ordinary differential equations.

# Statement of need

Turbulence down to kinetic scales is ubiquitous in the weakly collisional plasmas found in space, astrophysical, and laboratory environments, but resolving all relevant scales poses a significant computational challenge [@Schekochihin2009; @TenBarge2013; @Verscharen2019]. The widely used particle-in-cell (PIC) method suffers from small-scale noise due to the unrealistically low number of particles that is computationally feasible to run, and from inherent difficulties in conserving energy due to its discrete field–particle coupling [@Birdsall2005; @Markidis2011].

A frequently used alternative is to perform a spectral decomposition of the one-particle probability density function (PDF) and time-evolve a finite number of moments [@Roytershteyn2018; @Koshkarov2021; @Mandell2024] whose dynamical equations are derived from the Vlasov-Maxwell system. When the one-particle PDF is close to a Maxwellian distribution, it is appropriate to expand functions of the particle velocity into a truncated asymmetrically weighted Hermite basis, with the first three moments corresponding to a fluid model and subsequent moments adding kinetic corrections to it [@Delzanno2015]. This method is known to conserve particle number, linear momentum, and energy, and has been successfully implemented in the (Fortran-based) code Spectral Plasma Solver (SPS) [@Delzanno2015; @Vencels2016; @Roytershteyn2018], in which functions of the configuration-space coordinates are decomposed into Fourier modes, and in the more recent SPS-DG [@Koshkarov2021], which instead employs a discontinuous Galerkin discretization in configuration space. However, SPS and SPS-DG are not available as open-source software, limiting their accessibility, extensibility, and use in the broader research and educational community. With SPECTRAX we aim to fill this gap. It implements the same robust Hermite-Fourier spectral algorithm as SPS, but is built from the ground up in a modern, open-source, Python-based framework. By leveraging JAX [@Bradbury2018], SPECTRAX offers performance competitive with compiled languages while retaining the simplicity of Python. It is fully open-source, lowering the barrier for researchers to use, modify, and contribute to spectral Vlasov-Maxwell simulations.

# Structure

Schematically, the system of coupled ODEs solved in SPECTRAX looks as follows:
$$
\begin{aligned}
\frac{d\mathbf{C}}{dt}=&\mathcal{L}_1\mathbf{C}+\mathcal{N}(\mathbf{C},\mathbf{F}),\quad
    \frac{d\mathbf{F}}{dt}=&\mathcal{L}_2\mathbf{C}+\mathcal{L}_3\mathbf{F},
\end{aligned}
$$
where $\mathbf{C}$ and $\mathbf{F}$ are arrays containing the Hermite-Fourier coefficients of the one-particle PDF and the electromagnetic field, respectively, $\mathcal{L}_i$ with $i=1,2,3$ are linear operators, and $\mathcal{N}(\mathbf{C},\mathbf{F})$ is a nonlinear term. The truncation of the Hermite series leads to the well-known phenomenon of recursion, where the system returns to the initial state after a time that grows with the square root of the number of Hermite modes that are kept [@Canosa1974]. To suppress this unphysical behavior, we use a hypercollisional operator based on the Lenard-Bernstein operator. More details on the numerical method can be found in [@Delzanno2015; @Vencels2016; @Roytershteyn2018].

SPECTRAX is written entirely in Python and leverages the JAX library for just-in-time (JIT) compilation, automatic parallelization on a single core, and efficient execution on both CPUs and GPUs. The core logic is JIT-compiled with `@jit` decorators, ensuring that Python overhead is eliminated during the simulation loop. The code is organized into three main modules. At the top level, `_simulation.py` provides the main simulation driver and orchestrates the overall data flow. The simulation function initializes the full system state by calling `initialize_simulation_parameters` from `_initialization.py`, which supplies the initial conditions, grids, and pre-computed helper arrays. The system is then advanced in time using the Diffrax library [@Kidger2021]. During time integration, the `ode_system` function, defined in `_simulation.py` and invoked by the time integrator, assembles the time derivatives of all Hermite-Fourier coefficients for all plasma species, as well as the electric and magnetic fields. The evaluation of the right-hand side (RHS) of the coupled ODE system for the Hermite–Fourier moments of the one-particle PDF for each species is delegated to `_model.py`, where these equations are defined in the `Hermite_Fourier_system` function. Nonlinear terms are computed using a standard pseudo-spectral approach [@Patterson1971], in which a 2/3 de-aliasing mask is applied to the Hermite–Fourier coefficients of the distribution functions and fields, followed by Fourier anti-transforms, multiplication in real space, and Fourier transforms back to spectral space. All equations are assembled simultaneously using JAX array operations, allowing the full RHS to be evaluated efficiently in a single call. The plasma current entering the RHS of the electric field equations is computed via the `plasma_current` function, also defined in `_model.py`. The `_initialization.py` module provides the data required by both the simulation driver and the physics model. It handles the setup of simulation parameters, construction of all necessary grids (e.g., `kx_grid`), pre-computation of helper arrays, and loading of user-defined configuration through the `initialize_simulation_parameters` function.

At a high level, a SPECTRAX simulation proceeds as follows. A user-created Python script loads the simulation parameters from a TOML file (or alternatively defines them directly in the script), defines arrays with the initial Hermite-Fourier coefficients of all plasma species and the electric and magnetic fields, and then calls the `simulation` function. The initial conditions are then flattened and packed into a single 1D array to prepare them for Diffrax. Arrays with the time derivatives of all Hermite-Fourier coefficients for the plasma species and electric and magnetic fields are defined, flattened, and packed into a single 1D array to prepare them for Diffrax. The resulting ODE system is integrated in time using a Diffrax solver [@Kidger2021].

# Capabilities

SPECTRAX can simulate the time evolution of the full, self-consistent Vlasov-Maxwell system for two or more plasma species. Since functions of space coordinates are decomposed into Fourier modes, simulations naturally assume periodic boundary conditions in all spatial directions. The equations of motion for the Hermite-Fourier coefficients were derived assuming Cartesian coordinates in the phase space [@Delzanno2015; @Vencels2016; @Roytershteyn2018]. While SPECTRAX approximates the one-particle PDF in the fully six-dimensional phase space, lower-dimensional simulations are supported in the following sense. If in configuration space the number of Fourier modes along direction $j$ is set to 1, only the $k_j=0$ mode will be present, so all partial derivatives along that direction are zero, i.e., functions of configuration space coordinates will not depend on $j$. In velocity space, initializing a single Hermite mode along direction $v_j$ implies that, if the distribution function is factorized as three separate functions with each depending only on the particle velocity along one direction, the factor depending on $v_j$ would be a Maxwellian distribution with constant drift and thermal velocities. Thus, in a one-dimensional simulation in the $x$-direction, configuration space functions only depend on $x$, and velocity space functions are fixed Maxwellians along $v_y$ and $v_z$, having higher order Hermite corrections only along $v_x$; vector fields retain all three components.

The code has been verified against several standard kinetic plasma physics benchmarks, demonstrating its ability to capture key physical phenomena. First, we test 1D Landau damping, where a small sinusoidal perturbation to the electron density initializes a longitudinal electric wave. The left panel of \autoref{fig:landau} shows the linear decay of the electric energy, until the slower ion dynamics comes to dominate at late times, as expected. In the same panel, the measured damping rate is observed to closely match the analytical prediction.

The next benchmark is the 1D two-stream instability, where a small sinusoidal perturbation is initialized onto two separate electron populations with the same temperature but opposite drift velocities. As shown in the left panel of \autoref{fig:twostream}, the observed linear growth rate closely matches the analytical model.

Finally, we test SPECTRAX on the 2D Orszag–Tang vortex, a standard kinetic benchmark for plasma turbulence in which an initially smooth, large-scale flow and magnetic configuration evolves into a network of thin current sheets and multiscale magnetic fluctuations. We verify that SPECTRAX reproduces the characteristic out-of-plane current density reported in [@Vencels2016] (compare their Figure 1 to the left panel of \autoref{fig:orszag} here).

For details on the theoretical formulation and setup of these benchmark problems, see [@Delzanno2015; @Vencels2016; @Roytershteyn2018]. For a derivation of the linearized Vlasov-Poisson theory used to obtain the analytical damping and growth rates shown in \autoref{fig:landau} and \autoref{fig:twostream}, see [@Schekochihin2025]. The three benchmarks were run on a single CPU of a MacBook M4. Time stepping was performed with Diffrax's implementation of the Dormand-Prince 8/7 method. While we note that exact energy conservation is not expected from this method, excellent energy conservation was observed in these benchmarks (see \autoref{fig:landau} and \autoref{fig:twostream}, right panel).

The Orszag-Tang benchmark was also used to assess the computational performance of SPECTRAX on CPUs and GPUs. For this test, simulations were run on a single AMD EPYC 7763 CPU and a single NVIDIA A100 GPU, both on the
NERSC Perlmutter HPE Cray EX supercomputer. Simulations with varying numbers of Fourier and Hermite modes were executed and timed; the results are shown in \autoref{fig:orszag}, right panel. The use of a GPU consistently yields speedups of almost two orders of magnitude relative to a CPU. In addition, \autoref{tbl:compiletime} compares total runtimes, including JIT compilation time (corresponding to the first execution), with runtimes excluding compilation time (subsequent executions). The compilation overhead remains between 5 and 12 seconds across all tested configurations, and does not scale strongly with size.

| Hermite modes | 4 | 5 | 6 | 8 | 10 |
|---|---:|---:|---:|---:|---:|
| Wall-clock time (incl. compilation) (s) | 42 | 105 | 119 | 247 | 492 |
| Wall-clock time (excl. compilation) (s) | 33 | 100 | 107 | 242 | 486 |
| Compilation overhead (s) | 9 | 5 | 12 | 5 | 6 |

Table: Comparison of Orszag–Tang simulation runtime on GPU, with and without compilation time. \label{tbl:compiletime}


![1D linear Landau damping. Left: Electric energy and damping rate. Right: Energy vs time and relative energy error.\label{fig:landau}](figures/fig_landau.png)


![Two-stream instability. Left: Electric energy and growth rate. Right: Energy vs time and relative energy error.\label{fig:twostream}](figures/fig_twostream.png)


![Orszag–Tang vortex. Left: out-of-plane current density at $t\omega_{pe}=500$. Right: running time on CPU and GPU vs number of Hermite modes.\label{fig:orszag}](figures/fig_orszagtang.png)


# Acknowledgements

This work was supported by the National Science Foundation under Grants No. PHY-2409066 and PHY-2409316. This research used resources of the National Energy Research Scientific Computing Center, a DOE Office of Science User Facility supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231 using NERSC award NERSC DDR-ERCAP0030134.
