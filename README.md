# Seismic surface wave dispersion inversion

## Introduction

This package is for estimating 1D shear wave speed (Vs) profile from seismic Rayleigh wave phase velocity measurement at different periods. A Markov Chain Monte-Carlo method is deployed similar to previous study, but this package is totally object-oriented and allow more freedom in parameterization.

## Architecture
The inversion can be abstracted as finding a set of unknown variables, from which we can generate one 1D Vs Profile based on paramterization and satisfying Rayleigh wave observation. 

Variables -> 1D Vs Profile -> Phase Velocity -> modify variables based on fitness

The 1D Vs Profile, which is an instance of 1DModel class in this package, consists of multiple instances of the Layer class. And each Layer instance includes several instance of the BrownianVar class, which are to be estimated. A 3D model will be constructed by compiling all 1D profiles.

BrownianVar -> Layer -> 1DModel -> 3DModel

Point class contains both 1DModel and observations, and the method to estimate phase velocities. Postpoint is similar but include a method the keep only good enough models to estimate the posterior distribution

## Usage
Check synthetic_test() and realdata_test() in point.py for the usage.
