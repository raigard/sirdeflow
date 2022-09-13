"""
Title: Simplified Rheology Based Debris Flow Routing - SIRDEFLOW

Description: This code was developed for a master's dissertation. The code intends to simulate a debris flow mass
distribution over a slope with little information regarding the process. To simulate a debris flow it is required
at least a digital elevation model, a raster with initiation areas and failure depths, a kinematic viscosity (if
Newtonian approach is chosen). This code solves 1D velocities for different rheologies (Newtonian, Bingham plastic,
generalized Herschel-Bulkley). The direction of flow is determined by D8, Dinf of MFD algorithms. MFD is calculated
within SIRDEFLOW, whereas D8 and Dinf require a raster processed in other software - QGIS, SAGA, WhiteBoxTools, etc.
More information in Paul (2020) - 'Proposição de Modelo Para Simualação de Fluxos de Detritos em Escala de Bacia' in
Portuguese (PT-BR).

Author: Leonardo Rodolfo Paul

Contact: leonardorpaul@gmail.com

Last update: 09.13.2022

Version notes: i) Added the possibility to simulate for a specific number of iterations;


This code is intended to be free for non commercial applications (e.g. public, scientific and recreational applications)
If you intend to use this code under different scenarios, please contact the author.
"""
