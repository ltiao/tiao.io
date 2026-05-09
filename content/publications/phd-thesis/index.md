---
aliases:
  - /publication/phd-thesis/
title: "Probabilistic Machine Learning in the Age of Deep Learning: New Perspectives for Gaussian Processes, Bayesian Optimization and Beyond (PhD Thesis)"
authors:
- me
date: "2023-09-01T00:00:00Z"
doi: ""

# Schedule page publish date (NOT publication's date).
publishDate: "2017-01-01T00:00:00Z"

# Publication type.
# Accepts a single type but formatted as a YAML list (for Hugo requirements).
# Enter a publication type from the CSL standard.
publication_types: ["thesis"]

# Publication name and optional abbreviated publication name.
# publication: ""
# publication_short: ""

abstract: Advances in artificial intelligence (AI) are rapidly transforming our world, with systems now matching or surpassing human capabilities in areas ranging from game-playing to scientific discovery. Much of this progress traces back to machine learning (ML), particularly deep learning and its ability to uncover meaningful patterns and representations in data. However, true intelligence in AI demands more than raw predictive power; it requires a principled approach to making decisions under uncertainty. This highlights the necessity of probabilistic ML, which offers a systematic framework for reasoning about the unknown through probability theory and Bayesian inference. Gaussian processes (GPs) stand out as a quintessential probabilistic model, offering flexibility, data efficiency, and well-calibrated uncertainty estimates. They are integral to many sequential decision-making algorithms, notably Bayesian optimisation (BO), which has emerged as an indispensable tool for optimising expensive and complex black-box objective functions. While considerable efforts have focused on improving GP scalability, performance gaps persist in practice when compared against neural networks (NNs) due in large to its lack of representation learning capabilities. This, among other natural deficiencies of GPs, has hampered the capacity of BO to address critical real-world optimisation challenges. This thesis aims to unlock the potential of deep learning within probabilistic methods and reciprocally lend probabilistic perspectives to deep learning. The contributions include improving approximations to bridge the gap between GPs and NNs, providing a new formulation of BO that seamlessly accommodates deep learning methods to tackle complex optimisation problems, as well as a probabilistic interpretation of a powerful class of deep generative models for image style transfer. By enriching the interplay between deep learning and probabilistic ML, this thesis advances the foundations of AI and facilitates the development of more capable and dependable automated decision-making systems.


# Summary. An optional shortened abstract.
summary: This thesis explores the intersection of deep learning and probabilistic machine learning to enhance the capabilities of artificial intelligence. It addresses the limitations of Gaussian processes (GPs) in practical applications, particularly in comparison to neural networks (NNs), and proposes advancements such as improved approximations and a novel formulation of Bayesian optimization (BO) that seamlessly integrates deep learning methods. The contributions aim to enrich the interplay between deep learning and probabilistic ML, advancing the foundations of AI and fostering the development of more capable and reliable automated decision-making systems.

tags:
- Probabilistic Models
- Gaussian Processes
- Bayesian Optimization
- Variational Inference
- Deep Learning
- Machine Learning

featured: true


# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  focal_point: Center
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
# projects:
# - internal-project

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
# slides: example

links:
- name: Preprint
  url: phd-thesis-louis-tiao.pdf
- name: Full Acknowledgements
  url: post/phd-thesis-acknowledgements/
- type: pdf
  url: "https://hdl.handle.net/2123/32803"
---

The full text is available as a single PDF file <a href="publication/phd-thesis/phd-thesis-louis-tiao.pdf" target="_blank" rel="noopener">{{< icon name="download" >}}</a>

You can also find a list of contents and PDFs corresponding to each individual chapter below:

### Table of Contents

- Chapter 1: Introduction <a href="publication/phd-thesis/contents/1 Introduction.pdf" target="_blank" rel="noopener">{{< icon name="download" >}}</a>
- Chapter 2: Background <a href="publication/phd-thesis/contents/2 Background.pdf" target="_blank" rel="noopener">{{< icon name="download" >}}</a>
- Chapter 3: Orthogonally-Decoupled Sparse Gaussian Processes with Spherical Neural Network Activation Features <a href="publication/phd-thesis/contents/3 Orthogonally-Decoupled Sparse Gaussian Processes with Spherical Neural Network Activation Features.pdf" target="_blank" rel="noopener">{{< icon name="download" >}}</a>
- Chapter 4: Cycle-Consistent Generative Adversarial Networks as a Bayesian Approximation <a href="publication/phd-thesis/contents/4 Cycle-Consistent Generative Adversarial Networks as a Bayesian Approximation.pdf" target="_blank" rel="noopener">{{< icon name="download" >}}</a>
- Chapter 5: Bayesian Optimisation by Classification with Deep Learning and Beyond <a href="publication/phd-thesis/contents/5 Bayesian Optimisation by Classification with Deep Learning and Beyond.pdf" target="_blank" rel="noopener">{{< icon name="download" >}}</a>
- Chapter 6: Conclusion <a href="publication/phd-thesis/contents/6 Conclusion.pdf" target="_blank" rel="noopener">{{< icon name="download" >}}</a>
- Appendix A: Numerical Methods for Improved Decoupled Sampling of Gaussian Processes <a href="publication/phd-thesis/contents/A Numerical Methods for Improved Decoupled Sampling of Gaussian Processes.pdf" target="_blank" rel="noopener">{{< icon name="download" >}}</a>
- Bibliography <a href="publication/phd-thesis/contents/Bibliography.pdf" target="_blank" rel="noopener">{{< icon name="download" >}}</a>

Please find *Chapter 1: Introduction* reproduced in full below:

## Introduction

<!-- 
{{
Create your slides in Markdown - click the *Slides* button to check out the example.
{{

Add the publication's **full text** or **supplementary notes** here. You can use rich formatting such as including [code, math, and images](https://wowchemy.com/docs/content/writing-markdown-latex/).

## Introduction

- 
 -->
