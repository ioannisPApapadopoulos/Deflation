# Deflation

This repository contains untested codes in MATLAB, Julia, and Python for the implementation of the deflation algorithm for systematically computing multiple solutions of nonlinear systems (such as discretised PDEs).

For a description of the algorithm and its justification to infinite-dimensional problems, please read https://doi.org/10.1137/140984798 and https://doi.org/10.1080/10556788.2019.1613655 

On a computational level, deflation is very easy to implement. One solves the original _undeflated_ Newton/active-set system and simply multiplies the Newton update with a scalar (denoted by tau throughout the code). The scalar tau depends explicitly on the current Newton iterate and the Newton update of the undeflated Newton system. For the formula, checkout Eq. 3.23 in https://doi.org/10.1137/20M1326209

In the parent directory `self_contained_bratu.jl` is a self-contained Julia script for finding two solutions of a finite difference discretisation of the 1D Bratu equation from the same initial guess via deflation. The deflation operator is implemented in 2 lines! The `examples/` folder contains more examples using a slightly more sophisticated backend found in `src/`

There also codes for various active-set strategies for solving nonlinear systems that involve box constraints. 

## Other software

For more heavy-weight software that uses deflation, checkout:

1. defcon for bifurcation analysis (https://bitbucket.org/pefarrell/defcon) 
2. fir3dab for multiple solutions of 3D topology optimisation problems (https://github.com/ioannisPApapadopoulos/fir3dab)
3. BifurcationKit.jl for bifurcation analysis (https://github.com/bifurcationkit/BifurcationKit.jl)

## References

If you use code found in this repository, I would be grateful if you cited:

1. Farrell, Birkisson and Funke (2015) https://doi.org/10.1137/140984798
2. Papadopoulos, Farrell, Surowiec (2021) https://doi.org/10.1137/20M1326209
3. Papadopoulos, Farrell (2022) https://arxiv.org/abs/2202.08248

Moreover, if it is used with conjuction with an active-set solver as found in this repository:

4. Farrell, Croci, Surowiec (2020) https://doi.org/10.1080/10556788.2019.1613655

If one uses either of the Benson-Munson active-set strategies:

5. Benson, Munson (2006) https://doi.org/10.1080/10556780500065382

..or if one uses the primal-dual active set strategy (HIK) as implemented in this repository:

6. Hintermüller, Ito, Kunisch (2002) https://doi.org/10.1137/S1052623401383558

## DISCLAIMER

These scripts are untested and may contain bugs. Please raise a GitHub issue if you find any :)

Not everything is implemented in each of the three programming languages. If you have any requests, please raise a Github issue and I will be happy to help.

## CONTACT
ioannis.papadopoulos13@imperial.ac.uk
