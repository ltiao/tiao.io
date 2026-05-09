---
aliases:
  - /publication/spherical-features-gaussian-process/
title: 'Spherical Inducing Features for Orthogonally-Decoupled Gaussian Processes'

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - me
  - Vincent Dutordoir
  - Victor Picheny

date: '2023-04-25T00:00:00Z'
doi: ''
math: true

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ['paper-conference']

# Publication name and optional abbreviated publication name.
publication: Proceedings of the 40th International Conference on Machine Learning (ICML2023)
publication_short: In *ICML2023*. Accepted as *Oral Presentation*

abstract: Despite their many desirable properties, Gaussian processes (GPs) are often compared unfavorably to deep neural networks (NNs) for lacking the ability to learn representations. Recent efforts to bridge the gap between GPs and deep NNs have yielded a new class of inter-domain variational GPs in which the inducing variables correspond to hidden units of a feedforward NN. In this work, we examine some practical issues associated with this approach and propose an extension that leverages the orthogonal decomposition of GPs to mitigate these limitations. In particular, we introduce spherical inter-domain features to construct more flexible data-dependent basis functions for both the principal and orthogonal components of the GP approximation and show that incorporating NN activation features under this framework not only alleviates these shortcomings but is more scalable than alternative strategies. Experiments on multiple benchmark datasets demonstrate the effectiveness of our approach.

# Summary. An optional shortened abstract.
summary: We introduce spherical inter-domain inducing features that yield more flexible, data-dependent basis functions for orthogonally-decoupled GP approximations, narrowing the representation gap with neural networks while remaining scalable.

tags:
  - Machine Learning
  - Probabilistic Models
  - Gaussian Processes
  - Variational Inference

# Display this page in the Featured widget?
featured: true

# Custom links (uncomment lines below)
# links:
# - name: Custom Link
#   url: http://example.org


# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  placement: 1
  caption: '**Left:** A single-layer feedforward neural network on the unit sphere projected onto a plane in 3D; **Right:** Decoupling of a GP as a sum of orthogonal GPs'
  focal_point: Center
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects: []

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: ""

links:
- name: Conference Proceeding
  url: https://proceedings.mlr.press/v202/tiao23a.html
- type: pdf
  url: "https://arxiv.org/abs/2304.14034"
- type: code
  url: "https://github.com/ltiao/spherical-orthogonal-gaussian-processes"
- type: poster
  url: "poster.pdf"
- type: slides
  url: "https://docs.google.com/presentation/d/14Gf2pmvotLVpKD7k3JQFTGV-ywUpuVaHf5wm1InLd-U/edit?usp=sharing"
---
