---
featured: true
aliases:
  - /project/bore/
title: BORE
summary: A framework for Bayesian Optimization by probabilistic classification
tags:
  - Bayesian Optimization
  - Density Ratio Estimation
  - Hyperparameter Optimization
  - Open Source
  - AutoML
date: '2021-07-01T00:00:00Z'
draft: false

external_link: ''

image:
  caption: ''
  focal_point: Center

links:
  - icon: brands/github
    name: GitHub
    url: https://github.com/ltiao/bore
  - icon: hero/document-text
    name: ICML 2021 paper
    url: http://proceedings.mlr.press/v139/tiao21a.html
---

[BORE](https://github.com/ltiao/bore) is the reference implementation of
[*Bayesian Optimization by Density-Ratio Estimation*]({{< relref "/publications/bore-2" >}})
(Tiao et al., ICML 2021). It recasts the acquisition function in [Bayesian
optimization](/tags/bayesian-optimization/) as a probabilistic classification
problem via [density-ratio estimation](/tags/density-ratio-estimation/),
sidestepping the analytical-tractability constraints of conventional
surrogate-based methods.

Developed with Aaron Klein.
