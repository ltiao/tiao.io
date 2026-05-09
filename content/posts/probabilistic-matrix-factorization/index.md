---
aliases:
  - /post/probabilistic-matrix-factorization/
title: 'Link Prediction via Probabilistic Matrix Factorization with TensorFlow Probability'
summary: Probabilistic matrix factorization is a latent feature model that is used for link prediction and collaborative filtering.
authors:
- me
tags:
- Link Prediction
- TensorFlow Probability
date: "2019-03-22T00:00:00Z"
featured: false
draft: true

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Placement options: 1 = Full column width, 2 = Out-set, 3 = Screen-width
# Focal point options: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight
image:
  placement: 2
  caption: 'Interaction network containing conserved and essential protein complexes in *Escherichia coli* (Butland et al., 2005).'
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

{{< figure src="masked_adjacency_lower.png" caption="Adjacency Matrix (Lower Triangular Entries)." >}}

Test[^butland2005interaction]

[^butland2005interaction]: Butland, G., Peregrín-Alvarez, J. M., Li, J., Yang, W., Yang, X., Canadien, V., ... & Davey, M. (2005). Interaction network containing conserved and essential protein complexes in  *Escherichia coli*. Nature, 433(7025), 531.
