---
aliases:
  - /publication/empirical-gaussian-processes/
title: "Empirical Gaussian Processes"
authors:
  - Jihao Andreas Lin
  - Sebastian Ament
  - me
  - David Eriksson
  - Maximilian Balandat
  - Eytan Bakshy

author_notes:
  - 'Co-first author'
  - 'Co-first author'

date: '2026-02-01T00:00:00Z'
doi: ''

publication_types: ['paper-conference']
publication: Proceedings of the 43rd International Conference on Machine Learning (ICML 2026)
publication_short: In *ICML 2026*

abstract: |
  Gaussian processes (GPs) are powerful and widely used probabilistic
  regression models, but their effectiveness in practice is often limited by
  the choice of kernel function. This kernel function is typically
  handcrafted from a small set of standard functions, a process that requires
  expert knowledge, results in limited adaptivity to data, and imposes strong
  assumptions on the hypothesis space. We study Empirical GPs, a principled
  framework for constructing flexible, data-driven GP priors that overcome
  these limitations. Rather than relying on standard parametric kernels, we
  estimate the mean and covariance functions empirically from a corpus of
  historical observations, enabling the prior to reflect rich, non-trivial
  covariance structures present in the data. Theoretically, we show that the
  resulting model converges to the GP that is closest (in KL-divergence
  sense) to the real data generating process. Practically, we formulate the
  problem of learning the GP prior from independent datasets as likelihood
  estimation and derive an Expectation-Maximization algorithm with
  closed-form updates, allowing the model to handle heterogeneous observation
  locations across datasets. We demonstrate that Empirical GPs achieve
  competitive performance on learning curve extrapolation and time series
  forecasting benchmarks.

summary: |
  We study Empirical GPs, a principled framework for constructing flexible,
  data-driven Gaussian process priors. By estimating mean and covariance
  directly from a corpus of historical observations, we recover handcrafted
  kernels and outperform them on learning-curve extrapolation and time-series
  forecasting benchmarks.

tags:
  - Gaussian Processes
  - Bayesian Optimization
  - Probabilistic Models
  - Machine Learning

featured: true

image:
  caption: ''
  focal_point: Center
  preview_only: false

projects: []

slides: ""

links:
  - type: pdf
    url: "https://arxiv.org/abs/2602.12082"
---
