# NUTS_protein_superposition
Code for the Bioinformatics project of 7.5 ECTS, 'Bayesian protein superposition using NUTS algorithm'.

Protein superposition inference using Pyro, a Probabilistic Programming Language (PPL) developed by Uber Labs. Pytorch is used as the back-end. The inference is performed with the 'No U-Turn Sampler' algorithm, an improved and automatized version of a Hamiltonian Monte Carlo (HMC), a special type of Markov Chain Monte Carlo (MCMC).

- bayes_superposition_pyro_init.py is the code in which the NUTS kernel is initialized with Pyro's initialize_model()
- bayes_superposition_svi_init.py is the code in which the NUTS kernel is initialized with Lys Sanz Moreta's SVI function.
