---
aliases:
  - /publication/batch-bore-guarantees/
title: 'Batch Bayesian Optimisation via Density-ratio Estimation with Guarantees'
# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
- Rafael Oliveira
- me
- Fabio Ramos

date: '2022-12-01T00:00:00Z'
doi: ''

# Schedule page publish date (NOT publication's date).
publishDate: '2022-09-01T00:00:00Z'

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ['paper-conference']

# Publication name and optional abbreviated publication name.
publication: Advances in Neural Information Processing Systems 35 (NeurIPS2022)
publication_short: In *NeurIPS2022*

abstract: We extend BORE — a Bayesian optimization framework that recasts the acquisition function as a probabilistic classification problem via density-ratio estimation — to the batch setting, where multiple candidates are evaluated in parallel. We characterize the conditions under which the resulting algorithm enjoys theoretical convergence guarantees and demonstrate its practical effectiveness on a range of black-box optimization benchmarks.

# Summary. An optional shortened abstract.
summary: A batch extension of BORE with theoretical convergence guarantees for parallel Bayesian optimization.

tags:
  - Machine Learning
  - Density Ratio Estimation
  - Bayesian Optimization
  - Probabilistic Models
  - AutoML
  - Hyperparameter Optimization
# Display this page in the Featured widget?
featured: false


# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  caption: ''
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
- type: pdf
  url: "https://arxiv.org/abs/2209.10715"
- type: code
  url: "https://github.com/rafaol/batch-bore-with-guarantees"
---
