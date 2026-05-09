---
aliases:
  - /publication/vi-gcn-2/
title: 'Variational Inference for Graph Convolutional Networks in the Absence of Graph Data and Adversarial Settings'
# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
- Pantelis Elinas
- Edwin V. Bonilla
- me
# Author notes (optional)
author_notes:
  - 'Equal contribution'
  - 'Equal contribution'

date: '2020-06-01T00:00:00Z'
doi: ''

# Schedule page publish date (NOT publication's date).
# publishDate: '2017-01-01T00:00:00Z'

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ['paper-conference']

# Publication name and optional abbreviated publication name.
publication: Advances in Neural Information Processing Systems 33 (NeurIPS2020)
publication_short: In *NeurIPS2020*. Accepted as *Spotlight Presentation* (Awarded to Top 3% of Papers)

abstract: We propose a framework that lifts the capabilities of graph convolutional networks (GCNs) to scenarios where no input graph is given and increases their robustness to adversarial attacks. We formulate a joint probabilistic model that considers a prior distribution over graphs along with a GCN-based likelihood and develop a stochastic variational inference algorithm to estimate the graph posterior and the GCN parameters jointly. To address the problem of propagating gradients through latent variables drawn from discrete distributions, we use their continuous relaxations known as Concrete distributions. We show that, on real datasets, our approach can outperform state-of-the-art Bayesian and non-Bayesian graph neural network algorithms on the task of semi-supervised classification in the absence of graph data and when the network structure is subjected to adversarial perturbations.

# Summary. An optional shortened abstract.
summary: Our proposed framework uses a joint probabilistic model and stochastic variational inference to improve the performance and robustness of graph convolutional networks (GCNs) in scenarios without input graph data, outperforming state-of-the-art algorithms on semi-supervised classification tasks.

tags:
- Machine Learning
- Graph Representation Learning
- Semi-supervised Learning
- Probabilistic Models
- Variational Inference
# Display this page in the Featured widget?
featured: true


# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  caption: '**Left:** Observed graph. Edges of original graph are denoted by *solid lines*, while spuriously added edges are denoted by *maroon dashed lines*. **Right:** Resulting posterior probabilities over edges denoted by edge color opacity. With few exceptions, the posterior probabilities of the added edges are attenuated.'
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
  url: "https://papers.nips.cc/paper/2020/hash/d882050bb9eeba930974f596931be527-Abstract.html"
- type: code
  url: "https://github.com/ebonilla/VGCN"
- type: dataset
  url: "https://github.com/wowchemy/wowchemy-hugo-themes"
- type: video
  url: "https://slideslive.com/38937946"
---

This paper is a follow-up to our [working paper](../vi-gcn-1), previously
presented at the NeurIPS2019 Graph Representation Learning Workshop, now with 
significantly expanded experimental analyses.

<div id="presentation-embed-38937946"></div>
<script src='https://slideslive.com/embed_presentation.js'></script>
<script>
    embed = new SlidesLiveEmbed('presentation-embed-38937946', {
        presentationId: '38937946',
        autoPlay: false, // change to true to autoplay the embedded presentation
        verticalEnabled: true
    });
</script>
