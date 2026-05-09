---
aliases:
  - /publication/cycle-bayes/
title: "Cycle-Consistent Adversarial Learning as Approximate Bayesian Inference"
authors:
- me
- Edwin V. Bonilla
- Fabio Ramos
date: "2018-07-01T00:00:00Z"
doi: ""

# Schedule page publish date (NOT publication's date).
# publishDate: "2017-01-01T00:00:00Z"

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["article"]

# Publication name and optional abbreviated publication name.
publication: Presented at the ICML2018 Workshop on Theoretical Foundations and Applications of Deep Generative Models. Stockholm, Sweden, 2018.
publication_short: In *ICML2018* Theoretical Foundations and Applications of Deep Generative Models. Accepted as *Contributed Talk*.

abstract: We formalize the problem of learning interdomain correspondences in the absence of paired data as Bayesian inference in a latent variable model (LVM), where one seeks the underlying hidden representations of entities from one domain as entities from the other domain. First, we introduce implicit latent variable models, where the prior over hidden representations can be specified flexibly as an implicit distribution. Next, we develop a new variational inference (VI) algorithm for this model based on minimization of the symmetric Kullback-Leibler (KL) divergence between a variational joint and the exact joint distribution. Lastly, we demonstrate that the state-of-the-art cycle-consistent adversarial learning (CYCLEGAN) models can be derived as a special case within our proposed VI framework, thus establishing its connection to approximate Bayesian inference methods.

# Summary. An optional shortened abstract.
summary: We derive cycle-consistent adversarial learning (CycleGAN) as a special case of variational inference in a latent-variable model with implicit priors, establishing a Bayesian foundation for unsupervised inter-domain translation.

tags:
- Generative Adversarial Networks
- Generative Models
- Unsupervised Learning
- Probabilistic Models
- Latent Variable Models
- Variational Inference
- Density Ratio Estimation
- Machine Learning
featured: false


# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  caption: Norstedts Building on Riddarholmen in Stockholm, Sweden. &copy; Louis Tiao
  focal_point: Center
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects:
- implicit-models

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
slides: ""

links:
- name: Workshop Homepage
  url: https://sites.google.com/view/tadgm/home
- type: pdf
  url: "https://arxiv.org/abs/1806.01771"
- type: code
  url: "https://github.com/ltiao/cycle-bayes"
- type: poster
  url: "poster.pdf"
- type: slides
  url: "slides.pdf"
---
