# Floquet Topological Insulator Simulation

This repository contains a Python script for simulating and analyzing Floquet topological insulators, focusing on the anomalous and Haldane phases. The code calculates quasienergy spectra and visualizes edge state propagation in these systems.

## Purpose

The main objectives of this code are:

1. To construct and diagonalize Hamiltonians for different bond types in a lattice structure.
2. To calculate the time-evolution operator and quasienergy spectrum for both anomalous and Haldane phases.
3. To visualize the quasienergy spectrum as a function of momentum.
4. To simulate and visualize the time evolution of edge states in both phases.

## Methods

The simulation uses the following key methods:

### 1. Hamiltonian Construction

- `HA(n, Lx, ky)`: Constructs the Hamiltonian for A bonds.
- `HB(n, Lx, ky)`: Constructs the Hamiltonian for B bonds.
- `HC(n, Lx)`: Constructs the Hamiltonian for C bonds.

These functions create matrices representing the hopping terms between different sites in the lattice.

### 2. Time Evolution

- `U(n, Lx, T, ky)`: Calculates the time-evolution operator for a full period of the drive.

### 3. Eigenvalue Decomposition

- `eigu(U)`: A function provided by Kolodrubetz to diagonalize unitary matrices.

### 4. Quasienergy Calculation

The code calculates quasienergies by diagonalizing the time-evolution operator for different momentum values.

### 5. Edge State Propagation

The simulation evolves an initial state localized at one edge of the system and calculates the probability density of finding the particle at different positions over time.

## Visualization

The code generates several plots:

1. Quasienergy spectrum vs. momentum for the anomalous phase.
2. Quasienergy spectrum vs. momentum for the Haldane phase.
3. Time evolution of edge state propagation for the anomalous phase.
4. Time evolution of edge state propagation for the Haldane phase.

## Usage

To run the simulation:

1. Ensure you have the required dependencies installed (NumPy, Matplotlib).
2. Adjust the global variables at the top of the script if needed.
3. Run the script. It will generate and display the plots automatically.

## Parameters

Key parameters that can be adjusted include:

- `n`: Number of unit lattices
- `Lx`, `Ly`: Number of lattice sites in x and y directions
- `J`, `Jprime`: Hopping coefficients
- `T_A`, `T_H`: Driving periods for anomalous and Haldane phases
