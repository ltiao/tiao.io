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
summary: We explore the intersection of deep learning and probabilistic machine learning, addressing limitations of Gaussian processes in comparison to neural networks and proposing advancements such as improved approximations and a novel formulation of Bayesian optimization that seamlessly integrates deep-learning methods.

tags:
- Probabilistic Models
- Gaussian Processes
- Bayesian Optimization
- Variational Inference
- Deep Learning
- Machine Learning

featured: false


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
- type: pdf
  url: phd-thesis-louis-tiao.pdf
- type: source
  label: USyd Library
  url: "https://hdl.handle.net/2123/32803"
- type: source
  label: Full Acknowledgements
  url: /posts/phd-thesis-acknowledgements/
---

The full text is available as a single PDF file <a href="phd-thesis-louis-tiao.pdf" target="_blank" rel="noopener">{{< icon name="arrow-down-tray" >}}</a>

You can also find a list of contents and PDFs corresponding to each individual chapter below:

### Table of Contents

- Chapter 1: Introduction <a href="contents/1 Introduction.pdf" target="_blank" rel="noopener">{{< icon name="arrow-down-tray" >}}</a>
- Chapter 2: Background <a href="contents/2 Background.pdf" target="_blank" rel="noopener">{{< icon name="arrow-down-tray" >}}</a>
- Chapter 3: Orthogonally-Decoupled Sparse Gaussian Processes with Spherical Neural Network Activation Features <a href="contents/3 Orthogonally-Decoupled Sparse Gaussian Processes with Spherical Neural Network Activation Features.pdf" target="_blank" rel="noopener">{{< icon name="arrow-down-tray" >}}</a>
- Chapter 4: Cycle-Consistent Generative Adversarial Networks as a Bayesian Approximation <a href="contents/4 Cycle-Consistent Generative Adversarial Networks as a Bayesian Approximation.pdf" target="_blank" rel="noopener">{{< icon name="arrow-down-tray" >}}</a>
- Chapter 5: Bayesian Optimisation by Classification with Deep Learning and Beyond <a href="contents/5 Bayesian Optimisation by Classification with Deep Learning and Beyond.pdf" target="_blank" rel="noopener">{{< icon name="arrow-down-tray" >}}</a>
- Chapter 6: Conclusion <a href="contents/6 Conclusion.pdf" target="_blank" rel="noopener">{{< icon name="arrow-down-tray" >}}</a>
- Appendix A: Numerical Methods for Improved Decoupled Sampling of Gaussian Processes <a href="contents/A Numerical Methods for Improved Decoupled Sampling of Gaussian Processes.pdf" target="_blank" rel="noopener">{{< icon name="arrow-down-tray" >}}</a>
- Bibliography <a href="contents/Bibliography.pdf" target="_blank" rel="noopener">{{< icon name="arrow-down-tray" >}}</a>

Please find *Chapter 1: Introduction* reproduced in full below:

### Introduction
Artificial intelligence (AI) stands poised to be among the most disruptive technologies of our era. The breakneck pace of recent AI advancements has been spearheaded by machine learning (ML), particularly the resurgence of *deep learning*. Deep learning is as old as the first general-purpose electronic computer; with roots tracing back to the 1940s and ’50s ([McCulloch & Pitts, 1943](#ref-mcculloch1943logical); [Rosenblatt, 1958](#ref-rosenblatt1958perceptron)), the revival of deep learning, beginning in the early 2010s, was catalysed by a series of breakthroughs that shattered previously perceived limitations and captivated the collective imagination. These breakthroughs span various domains, including computer vision ([Girshick et al., 2014](#ref-girshick2014rich); [Krizhevsky et al., 2012](#ref-krizhevsky2012imagenet); [Redmon et al., 2016](#ref-redmon2016you); [Ronneberger et al., 2015](#ref-ronneberger2015u)), speech recognition ([Graves et al., 2013](#ref-graves2013speech); [Hinton et al., 2012](#ref-hinton2012deep)), natural language processing ([Brown et al., 2020](#ref-brown2020language); [Vaswani et al., 2017](#ref-vaswani2017attention)), protein folding ([Jumper et al., 2021](#ref-jumper2021highly)), generative art and artificial creativity  ([Goodfellow et al., 2014](#ref-goodfellow2014generative); [Ho et al., 2020](#ref-ho2020denoising); [Ramesh et al., 2022](#ref-ramesh2022hierarchical); [Rombach et al., 2022](#ref-rombach2022high)), as well as reinforcement learning for robotics control ([Lillicrap et al., 2015](#ref-lillicrap2015continuous); [Mnih et al., 2015](#ref-mnih2015human)) and achieving superhuman-level gameplay ([Mnih et al., 2013](#ref-mnih2013playing); [Silver et al., 2016](#ref-silver2016mastering)).

Nevertheless, it is crucial to view these developments as means to an ultimate end rather than an end in themselves. Arguably, the true pinnacle of AI’s capabilities lies in optimal *decision-making*, whether that entails offering analyses and insights to aid humans in making better decisions or completely automating the decision-making process altogether. Practically any task directed towards a well-defined objective can be boiled down to a cascade of decisions. At a fundamental level, operating a vehicle involves a continuous stream of decisions involving accelerating, braking, and turning. Financial trading revolves around decisions to buy, sell, or hold various assets. Even complex engineering tasks, such as designing an aerofoil, involve a sequence of decisions about adjusting design variables to achieve desirable aerodynamic characteristics.

Yet, the intricacies of decision-making surpass what any single advancement in deep learning can address. While convolutional neural networks (CNNs) can facilitate object detection tasks in autonomous vehicles, recurrent neural networks (RNNs) can aid in forecasting market dynamics for systematic trading, and physics-informed NNs can assist in predicting aerodynamic effects, it remains the case that no target or quantity of interest can be entirely known or predictable (indeed, if they were, the pursuit of predictive modelling and ML would be superfluous). Instead, predictions often prove unreliable, or at best, *uncertain*, due to the limitations of our knowledge and the complexity and variability inherent in the underlying real-world processes. The impressive power of deep learning models often overshadows their ignorance of the limits of their own knowledge and the extent of uncertainty in their predictions. When these predictions are integrated into a sequential decision-making framework, such uncertainty can amplify, compound, and lead to catastrophic consequences. In the context of aeronautical engineering, this could result in inefficient designs; in quantitative finance, it can lead to devastating capital losses; and in autonomous driving, it can even cost lives.

#### Probabilistic Machine Learning

Grounded in the laws of probability and Bayesian statistics ([Bayes, 1763](#ref-bayes1763lii); [Laplace, 1814](#ref-laplace1814theorie)), *probabilistic* ML provides a consistent framework for systematically reasoning about the unknown. The probabilistic approach to ML acknowledges that the real world is fraught with uncertainty and embraces this uncertainty as an inherent part of decision-making. Unlike traditional methods, including those of deep learning, it recognises model predictions not as absolute truths that can be represented as single *point estimates* produced from a deterministic mapping, but as full *probability distributions* that capture the potential outcomes of a random variable as it propagates through some underlying data-generating process. In a *probabilistic model*, all quantities are treated as random variables governed by probability distributions – the data are treated as observed variables, which are influenced by some underlying hidden variables, e.g., the model parameters. A prior distribution is used to express reasonable values for these hidden variables and to eliminate implausible ones. The relationship between observed and hidden variables is described using the likelihood, and the process of Bayesian inference amounts to calculating, using basic laws of probability, a posterior distribution over the hidden factors conditioned on the observed data, which can be seen as a refinement of the prior beliefs in light of new evidence. While the posterior distribution can be useful in and of itself, its primary role lies in facilitating subsequent prediction and decision-making by providing full probability distributions over predicted outcomes. This capability allows the decision-maker to assess the range of possible scenarios and their associated probabilities, enabling a more nuanced understanding of uncertainty and risk, which is indispensable in complex, dynamic environments where the repercussions of incorrect decisions can be severe. In essence, probabilistic ML equips autonomous decision-making systems with a probabilistic worldview, enabling them to navigate ambiguity and make sound decisions in the face of imperfect information.

#### Probabilistic ML vs. Deep Learning

While deep learning has dominated recent AI advances, probabilistic ML remains as important as ever and continues to offer valuable tools for addressing AI challenges that can not be fully resolved by deep learning alone. Although both approaches can be combined to create hybrid methods that leverage their respective strengths, some defining characteristics have traditionally set deep learning apart from probabilistic ML. Perhaps most notably, probabilistic ML approaches can achieve remarkable predictive performance even when data is scarce. In contrast, deep learning models tend to be data-intensive by nature, often demanding datasets of a scale proportional to their size (i.e., their parameter count) ([Hoffmann et al., 2022](#ref-hoffmann2022training)), which has seen explosive growth in recent years ([Anil et al., 2023](#ref-anil2023palm); [OpenAI, 2023](#ref-openai2023gpt); [Rae et al., 2021](#ref-rae2021scaling); [Shoeybi et al., 2019](#ref-shoeybi2019megatron); [Touvron et al., 2023](#ref-touvron2023llama)). With that being said, inference in many probabilistic models poses computational problems that are difficult to scale. On the other hand, deep learning approaches have excelled in scalability, a key factor contributing to their widespread success. This scalability is bolstered by their compatibility with various speed-enhancing mechanisms such as stochastic optimisation, specialised hardware accelerators (GPUs and TPUs), as well as distributed and/or cloud-based computing infrastructure. To bridge this gap, substantial research effort has been devoted to enabling probabilistic ML to benefit from these advantages through optimisation-based approximations to Bayesian inference ([Jordan et al., 1998](#ref-jordan1998introduction)).

Moreover, as mentioned earlier, these paradigms are by no means mutually exclusive. Indeed, it is often possible to directly extend existing models with a Bayesian treatment of their parameters, adding a layer of probabilistic reasoning to the model, and allowing it to not only make predictions but also estimate the uncertainty associated with those predictions. An excellent example is the BNN, which treats the weights as hidden variables and leverages posterior inference to provide predictions while estimating associated uncertainties, delivering a more robust and principled approach to deep learning ([Blundell et al., 2015](#ref-blundell2015weight); [MacKay, 1992](#ref-mackay1992practical); [Neal, 1995](#ref-neal1995bayesian)).

The Bayesian formalism naturally gives rise to many popular methods and paradigms, often in the form of point estimates or other kinds of approximations. The quintessential example of this is found in linear regression, in particular, in ridge and lasso regression ([Tibshirani, 1996](#ref-tibshirani1996regression)), which correspond variously to maximum *a posteriori* (MAP) estimates in Bayesian linear regression (BLR) models with prior distributions possessing different sparsity-inducing characteristics ([Gelman et al., 2013](#ref-gelman2013bayesian)) – more broadly, mitigations against over-fitting tend to arise organically in Bayesian methods, which is why they are frequently characterised as being fundamentally more robust against over-fitting ([Rasmussen & Williams, 2005, Section 5.2](#ref-10.7551/mitpress/3206.001.0001)). Likewise, the once *à la mode* support vector machines (SVMs) can be seen as MAP estimates for a class of nonparametric Bayesian models ([Opper & Winther, 2000](#ref-opper2000gaussian)), dropout ([Srivastava et al., 2014](#ref-srivastava2014dropout)) in NNs can be seen as a variational approximation to exact inference in BNNs ([Gal & Ghahramani, 2016](#ref-gal2016dropout)), and unsupervised learning methods such as factor analysis (FA) ([Spearman, 1904](#ref-spearman1904general)) and principal component analysis (PCA) ([Pearson, 1901](#ref-pearson1901liii)) are instances of a class of LVMs ([Bartholomew et al., 2011](#ref-bartholomew2011latent); [Tipping & Bishop, 1999](#ref-tipping1999probabilistic)) known as linear-Gaussian factor models ([Roweis & Ghahramani, 1999](#ref-roweis1999unifying)), to name just a few examples. Time and again, classical approaches have not only benefitted from being viewed through the Bayesian perspective but have also been enriched and redefined by the depth of insights this framework provides.

### Thesis Goals

The over-arching goal of this thesis is to continue advancing the integration and cross-pollination between deep learning and probabilistic ML. We aim to further the interplay between these two fields, both by incorporating probabilistic interpretations and uncertainty quantification into popular deep learning frameworks, and by leveraging the representational power of deep NNs to improve established Bayesian methods. This dual-pronged approach provides fresh perspectives and taps the complementary strengths of both paradigms, advancing the foundations of AI and facilitating the development of more capable and dependable decision support frameworks. Ultimately, we strive to unlock the potential of deep learning within high-impact probabilistic ML methodologies, and to lend useful Bayesian perspectives on current deep learning techniques.

#### Gaussian Process Models

Arguably, no family of probabilistic models embodies the ethos of probabilistic ML and illustrates its nuances and parallels with deep learning quite like the GP. Accordingly, they shall occupy a prominent place in our thesis. In particular, GPs stand out as the ideal choice when dealing with limited data, offer the flexibility to encode prior beliefs through the covariance function, and provide predictive uncertainty estimates with a fine calibration that is second to none. Conversely, they are challenging to scale to large datasets, a limitation that has spurred extensive research and development efforts. Furthermore, in contrast to deep learning models, which are often lauded for their ability to automatically uncover valuable patterns and features in data, GPs have at times been dismissed as unsophisticated smoothing mechanisms ([MacKay, 2003](#ref-mackay2003information)). Despite these apparent disparities, GPs are intricately connected to NNs in numerous ways. Among these, one of the most classical and well-known relationships is the convergence of single-layer NNs with randomly initialised weights toward GPs in the infinite-width limit ([Neal, 1995](#ref-neal1995bayesian)). Similar links have also been identified between GPs and infinitely wide *deep* NNs ([Lee et al., 2017](#ref-lee2017deep); [Matthews et al., 2018](#ref-matthews2018gaussian)).

In an effort to elevate the representational capabilities of GPs to a level comparable with deep NNs, DGPs ([Damianou & Lawrence, 2013](#ref-damianou2013deep)) stack together multiple layers of GPs. Additional efforts to construct efficient sparse GP approximations have leveraged the advantageous properties of computations on the hypersphere ([Dutordoir et al., 2020](#ref-dutordoir2020sparse)), which has led to deep GP (DGP) models in which the propagation of posterior predictive means is equivalent to a forward pass through a deep neural network (NN) ([Dutordoir et al., 2021](#ref-dutordoir2021deep); [Sun et al., 2020](#ref-sun2020neural)). Notably, as a side effect, this model effectively provides uncertainty estimates for deep NN through its predictive variance. Among the contributions of our thesis is the further development of this framework, integrating cutting-edge techniques ([Salimbeni et al., 2018](#ref-salimbeni2018orthogonally); [Shi et al., 2020](#ref-shi2020sparse)) to address some of its practical limitations, thereby narrowing the performance gap between GPs and deep NNs.

Probabilistic models, serving a crucial role as decision support tools, routinely aid scientific discovery in fields such as physics and astronomy, guiding advancements in areas of medicine and healthcare encompassing bioinformatics, epidemiology, and medical diagnosis. Beyond that, these models have wide-ranging applications in economics, econometrics, and the social sciences. Moreover, they are indispensable in various engineering disciplines, such as robotics and environmental engineering. Among the many probabilistic models, GPs stand out as a powerful driving force behind a number of important sequential decision-making frameworks, including active learning ([Houlsby et al., 2011](#ref-houlsby2011bayesian)) and reinforcement learning ([Deisenroth & Rasmussen, 2011](#ref-deisenroth2011pilco)), and the broader area of probabilistic numerics at large ([Hennig et al., 2022](#ref-hennig2022probabilistic)). Notably, Bayesian optimisation (BO) ([Brochu et al., 2010](#ref-brochu2010tutorial); [Garnett, 2023](#ref-garnett_bayesoptbook_2023); [Shahriari et al., 2015](#ref-shahriari2015taking)) is one major area that relies heavily on GPs and will feature extensively in our thesis.

#### Bayesian optimisation

BO is a powerful methodology dedicated to the global optimisation of complex and resource-intensive objective functions. In contrast to classical optimisation methods, BO excels even when dealing with functions that lack strong assumptions or guarantees. These functions may not be convex, possess no gradients, lack a well-defined mathematical form, and observable only indirectly through noisy measurements.

At its core, BO is a sequential decision-making algorithm.

It relies on observations from past function evaluations to determine the next candidate location for evaluation in pursuit of optimal solutions. BO leverages a probabilistic model, often a GP, to represent its knowledge and beliefs about the unknown function. This model is continuously updated with the acquisition of each new observation, enabling the algorithm to adapt its behaviour and make sound decisions based on the evolving information.

BO effectively manages uncertainty inherent in such sequential decision-making processes by making use of the probabilistic model to the fullest, harnessing the entire predictive distribution, particularly, the predictive uncertainty, to select promising candidate solutions that bring the most value to the optimisation process. This generally consists not merely of those most likely to optimise the objective function (i.e., *exploiting* that which is known), but also those likely to reveal the most knowledge and information about the function itself (i.e., *exploring* that which remains unknown).

This pronounced emphasis on well-calibrated uncertainty distinguishes BO as one of the standout “killer apps” for GPs and a jewel in the crown of probabilistic ML applications. In practice, BO has proven instrumental across science, engineering, and industry, where efficiency and cost-effectiveness are paramount. Its applications include protein engineering ([Romero et al., 2013](#ref-romero2013navigating); [Yang et al., 2019](#ref-yang2019machine)), material discovery ([Seko et al., 2015](#ref-seko2015prediction)), experimental physics (e.g., experiments involving ultra-cold atoms ([Wigley et al., 2016](#ref-wigley2016fast)) and free-electron lasers ([Duris et al., 2020](#ref-duris2020bayesian))), environmental monitoring (sensor placement) ([Garnett et al., 2010](#ref-garnett2010bayesian); [Marchant & Ramos, 2012](#ref-marchant2012bayesian)), and the design of aerodynamic aerofoils ([Forrester & Keane, 2009](#ref-forrester2009recent); [Lam et al., 2018](#ref-lam2018advances)), integrated circuits ([Lyu et al., 2017](#ref-lyu2017efficient); [Torun et al., 2018](#ref-torun2018global)), broadband high-efficiency power amplifiers ([Chen et al., 2015](#ref-chen2015bayesian)), and fast-charging protocols for lithium-ion batteries ([Attia et al., 2020](#ref-attia2020closed)). Notably, it has played a crucial role in automating the hyperparameter tuning of various ML models ([Snoek et al., 2012](#ref-snoek2012practical); [Turner et al., 2021](#ref-turner2021bayesian)), especially deep learning models, thus representing yet another way in which probabilistic ML has contributed to the advancement of deep learning.

However, GPs are not universally suitable for all BO problem scenarios. They are most effective when dealing with smooth, stationary functions with homoscedastic noise and a relatively modest input dimensionality. Additionally, GPs are easiest to work with for functions with a single output and purely continuous inputs. While a surprisingly wide array of real-world challenges satisfy these conditions, many high-impact problems, such as gene and protein design, which involves sequential inputs ([Gonzalez et al., 2015](#ref-gonzalez2015bayesian); [Hie & Yang, 2022](#ref-hie2022adaptive); [Moss et al., 2020](#ref-moss2020boss); [Romero et al., 2013](#ref-romero2013navigating); [Yang et al., 2019](#ref-yang2019machine)); NAS, which involves structured inputs with intricate conditional dependencies; and automotive safety engineering, which involve numerous constraints and multiple objectives, clearly fall outside of this scope. This is not to say that GPs cannot be extended to such challenging scenarios. However, such extensions almost always come at a cost. Consequently, it makes sense to appeal to alternative modelling paradigms more naturally suited to specific tasks, e.g., employing random forests (RFs) to handle discrete and structured inputs, or deep NNs for capturing nonstationary behaviour and dealing with multiple objectives. A major contribution of this thesis is the introduction of a new formulation of BO that seamlessly accommodates virtually any modelling paradigm, including deep learning, without any compromise.

### Thesis Overview

The core contributions of our thesis are summarised as follows:

1.  <span id="item:contrib-orthogonal-sparse-spherical-gp" label="item:contrib-orthogonal-sparse-spherical-gp"></span> We improve upon the framework for sparse hyperspherical GP approximations that employ nonlinear activations as inter-domain inducing features. This framework serves as a bridge between GPs and NNs, with posterior predictive mean taking the form of single-layer feedforward NNs. Our thesis examines some practical issues associated with this approach and proposes an extension that takes advantage of the orthogonal decoupling of GPs to mitigate these limitations. In particular, we introduce spherical inter-domain features to construct more flexible data-dependent basis functions for both the principal and orthogonal components of the GP approximation. We demonstrate that incorporating orthogonal inducing variables under this framework not only alleviates these shortcomings but also offers superior scalability compared to alternative strategies.

2.  <span id="item:contrib-cycle-bayes" label="item:contrib-cycle-bayes"></span> We provide a probabilistic perspective on cycle-consistent adversarial networks (CYCLEGANs), a cutting-edge deep generative model for style transfer and image-to-image translation. Specifically, we frame the problem of learning cross-domain correspondences without paired data as Bayesian inference in a latent variable model (LVM), in which the goal is to uncover the hidden representations of entities from one domain as entities in another. First, we introduce implicit LVMs, which allow flexible prior specification over latent representations as implicit distributions. Next, we develop a new variational inference (VI) framework that minimises a symmetrised statistical divergence between the variational and true joint distributions. Finally, we show that CYCLEGANs emerge as a closely-related variant of our framework, providing a useful interpretation as a Bayesian approximation.

3.  <span id="item:contrib-bore" label="item:contrib-bore"></span> We introduce a model-agnostic formulation of BO based on classification. Building on the established links between class-probability estimation (CPE), density-ratio estimation (DRE), and the improvement-based acquisition functions, we reformulate the acquisition function as a binary classifier over candidate solutions. This approach eliminates the need for an explicit probabilistic model of the objective function and casts aside the limitations of tractability constraints. As a result, our model-agnostic BO approach substantially broadens its applicability across diverse problem scenarios, accommodating flexible and scalable modelling paradigms such as deep learning without necessitating approximations or sacrificing expressive and representational capacity.

Accordingly, our thesis is organised as follows:

- Chapter 2 (Background) lays the necessary groundwork for our thesis. We begin by outlining the fundamental principles of probability and Bayesian statistics, which form the basis of probabilistic ML. Additionally, we introduce the widely-adopted method of approximate Bayesian inference known as VI. Our discussion underscores the central role played by statistical divergences, prompting us to delve into a larger family of divergences and motivating our discussion of DRE. With a solid foundation in place, we shift our focus to GPs, providing an introductory overview and highlighting the most commonly-used sparse approximations. Finally, we conclude this background chapter by introducing the basic concepts behind BO.

- Chapter 3 (Orthogonally-Decoupled Sparse GPs with Spherical Inducing Features) examines orthogonally-decoupled sparse GPs with spherical NN activation features, as summarised in the corresponding item above.

- Chapter 4 (Cycle-Consistent Adversarial Learning as Bayesian Inference) examines from the perspective of approximate Bayesian inference, as summarised in the corresponding item above.

- Chapter 5 (Bayesian Optimization by Density-Ratio Estimation) examines our model-agnostic approach to BO based on binary classification and DRE, as summarised in the corresponding item above.

- Chapter 6 (Conclusion) brings this thesis to a close by reflecting on our main contributions and situating them in the broader landscape of probabilistic methods in ML. Finally, we conclude by presenting our outlook on the avenues for future research and development in this rapidly evolving field.

### References
<div id="refs" class="references csl-bib-body hanging-indent" entry-spacing="0" line-spacing="2">

<div id="ref-anil2023palm" class="csl-entry">

Anil, R., Dai, A. M., Firat, O., Johnson, M., Lepikhin, D., Passos, A., Shakeri, S., Taropa, E., Bailey, P., Chen, Z., et al. (2023). Palm 2 technical report. *arXiv Preprint arXiv:2305.10403*.

</div>

<div id="ref-attia2020closed" class="csl-entry">

Attia, P. M., Grover, A., Jin, N., Severson, K. A., Markov, T. M., Liao, Y.-H., Chen, M. H., Cheong, B., Perkins, N., Yang, Z., et al. (2020). Closed-loop optimization of fast-charging protocols for batteries with machine learning. *Nature*, *578*(7795), 397–402.

</div>

<div id="ref-bartholomew2011latent" class="csl-entry">

Bartholomew, D. J., Knott, M., & Moustaki, I. (2011). *Latent variable models and factor analysis: A unified approach*. John Wiley & Sons.

</div>

<div id="ref-bayes1763lii" class="csl-entry">

Bayes, T. (1763). LII. An essay towards solving a problem in the doctrine of chances. By the late rev. Mr. Bayes, FRS communicated by mr. Price, in a letter to john canton, AMFR s. *Philosophical Transactions of the Royal Society of London*, *53*, 370–418.

</div>

<div id="ref-blundell2015weight" class="csl-entry">

Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight uncertainty in neural network. *International Conference on Machine Learning*, 1613–1622.

</div>

<div id="ref-brochu2010tutorial" class="csl-entry">

Brochu, E., Cora, V. M., & De Freitas, N. (2010). A tutorial on bayesian optimization of expensive cost functions, with application to active user modeling and hierarchical reinforcement learning. *arXiv Preprint arXiv:1012.2599*.

</div>

<div id="ref-brown2020language" class="csl-entry">

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, *33*, 1877–1901.

</div>

<div id="ref-chen2015bayesian" class="csl-entry">

Chen, P., Merrick, B. M., & Brazil, T. J. (2015). Bayesian optimization for broadband high-efficiency power amplifier designs. *IEEE Transactions on Microwave Theory and Techniques*, *63*(12), 4263–4272.

</div>

<div id="ref-damianou2013deep" class="csl-entry">

Damianou, A., & Lawrence, N. D. (2013). Deep gaussian processes. *Artificial Intelligence and Statistics*, 207–215.

</div>

<div id="ref-deisenroth2011pilco" class="csl-entry">

Deisenroth, M., & Rasmussen, C. E. (2011). PILCO: A model-based and data-efficient approach to policy search. *Proceedings of the 28th International Conference on Machine Learning (ICML-11)*, 465–472.

</div>

<div id="ref-duris2020bayesian" class="csl-entry">

Duris, J., Kennedy, D., Hanuka, A., Shtalenkova, J., Edelen, A., Baxevanis, P., Egger, A., Cope, T., McIntire, M., Ermon, S., et al. (2020). Bayesian optimization of a free-electron laser. *Physical Review Letters*, *124*(12), 124801.

</div>

<div id="ref-dutordoir2020sparse" class="csl-entry">

Dutordoir, V., Durrande, N., & Hensman, J. (2020). Sparse Gaussian processes with spherical harmonic features. *International Conference on Machine Learning*, 2793–2802.

</div>

<div id="ref-dutordoir2021deep" class="csl-entry">

Dutordoir, V., Hensman, J., Wilk, M. van der, Ek, C. H., Ghahramani, Z., & Durrande, N. (2021). Deep neural networks as point estimates for deep Gaussian processes. *Advances in Neural Information Processing Systems*, *34*.

</div>

<div id="ref-forrester2009recent" class="csl-entry">

Forrester, A. I., & Keane, A. J. (2009). Recent advances in surrogate-based optimization. *Progress in Aerospace Sciences*, *45*(1-3), 50–79.

</div>

<div id="ref-gal2016dropout" class="csl-entry">

Gal, Y., & Ghahramani, Z. (2016). Dropout as a bayesian approximation: Representing model uncertainty in deep learning. *International Conference on Machine Learning*, 1050–1059.

</div>

<div id="ref-garnett_bayesoptbook_2023" class="csl-entry">

Garnett, R. (2023). *<span class="nocase">Bayesian Optimization</span>*. Cambridge University Press.

</div>

<div id="ref-garnett2010bayesian" class="csl-entry">

Garnett, R., Osborne, M. A., & Roberts, S. J. (2010). Bayesian optimization for sensor set selection. *Proceedings of the 9th ACM/IEEE International Conference on Information Processing in Sensor Networks*, 209–219.

</div>

<div id="ref-gelman2013bayesian" class="csl-entry">

Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian data analysis*. CRC press.

</div>

<div id="ref-girshick2014rich" class="csl-entry">

Girshick, R., Donahue, J., Darrell, T., & Malik, J. (2014). Rich feature hierarchies for accurate object detection and semantic segmentation. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 580–587.

</div>

<div id="ref-gonzalez2015bayesian" class="csl-entry">

Gonzalez, J., Longworth, J., James, D. C., & Lawrence, N. D. (2015). Bayesian optimization for synthetic gene design. *arXiv Preprint arXiv:1505.01627*.

</div>

<div id="ref-goodfellow2014generative" class="csl-entry">

Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative adversarial networks. *arXiv Preprint arXiv:1406.2661*.

</div>

<div id="ref-graves2013speech" class="csl-entry">

Graves, A., Mohamed, A., & Hinton, G. (2013). Speech recognition with deep recurrent neural networks. *2013 IEEE International Conference on Acoustics, Speech and Signal Processing*, 6645–6649.

</div>

<div id="ref-hennig2022probabilistic" class="csl-entry">

Hennig, P., Osborne, M. A., & Kersting, H. P. (2022). *Probabilistic numerics*. Cambridge University Press.

</div>

<div id="ref-hie2022adaptive" class="csl-entry">

Hie, B. L., & Yang, K. K. (2022). Adaptive machine learning for protein engineering. *Current Opinion in Structural Biology*, *72*, 145–152.

</div>

<div id="ref-hinton2012deep" class="csl-entry">

Hinton, G., Deng, L., Yu, D., Dahl, G. E., Mohamed, A., Jaitly, N., Senior, A., Vanhoucke, V., Nguyen, P., Sainath, T. N., et al. (2012). Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups. *IEEE Signal Processing Magazine*, *29*(6), 82–97.

</div>

<div id="ref-ho2020denoising" class="csl-entry">

Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems*, *33*, 6840–6851.

</div>

<div id="ref-hoffmann2022training" class="csl-entry">

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., Casas, D. de L., Hendricks, L. A., Welbl, J., Clark, A., et al. (2022). Training compute-optimal large language models. *arXiv Preprint arXiv:2203.15556*.

</div>

<div id="ref-houlsby2011bayesian" class="csl-entry">

Houlsby, N., Huszár, F., Ghahramani, Z., & Lengyel, M. (2011). Bayesian active learning for classification and preference learning. *arXiv Preprint arXiv:1112.5745*.

</div>

<div id="ref-jordan1998introduction" class="csl-entry">

Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1998). An introduction to variational methods for graphical models. *Learning in Graphical Models*, 105–161.

</div>

<div id="ref-jumper2021highly" class="csl-entry">

Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., Tunyasuvunakool, K., Bates, R., Žídek, A., Potapenko, A., et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, *596*(7873), 583–589.

</div>

<div id="ref-krizhevsky2012imagenet" class="csl-entry">

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems*, *25*.

</div>

<div id="ref-lam2018advances" class="csl-entry">

Lam, R., Poloczek, M., Frazier, P., & Willcox, K. E. (2018). Advances in bayesian optimization with applications in aerospace engineering. *2018 AIAA Non-Deterministic Approaches Conference*, 1656.

</div>

<div id="ref-laplace1814theorie" class="csl-entry">

Laplace, P. S. (1814). *Théorie analytique des probabilités*. Courcier.

</div>

<div id="ref-lee2017deep" class="csl-entry">

Lee, J., Bahri, Y., Novak, R., Schoenholz, S. S., Pennington, J., & Sohl-Dickstein, J. (2017). Deep neural networks as gaussian processes. *arXiv Preprint arXiv:1711.00165*.

</div>

<div id="ref-lillicrap2015continuous" class="csl-entry">

Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2015). Continuous control with deep reinforcement learning. *arXiv Preprint arXiv:1509.02971*.

</div>

<div id="ref-lyu2017efficient" class="csl-entry">

Lyu, W., Xue, P., Yang, F., Yan, C., Hong, Z., Zeng, X., & Zhou, D. (2017). An efficient bayesian optimization approach for automated optimization of analog circuits. *IEEE Transactions on Circuits and Systems I: Regular Papers*, *65*(6), 1954–1967.

</div>

<div id="ref-mackay1992practical" class="csl-entry">

MacKay, D. J. (1992). A practical bayesian framework for backpropagation networks. *Neural Computation*, *4*(3), 448–472.

</div>

<div id="ref-mackay2003information" class="csl-entry">

MacKay, D. J. (2003). *Information theory, inference and learning algorithms*. Cambridge university press.

</div>

<div id="ref-marchant2012bayesian" class="csl-entry">

Marchant, R., & Ramos, F. (2012). Bayesian optimisation for intelligent environmental monitoring. *2012 IEEE/RSJ International Conference on Intelligent Robots and Systems*, 2242–2249.

</div>

<div id="ref-matthews2018gaussian" class="csl-entry">

Matthews, A. G. de G., Rowland, M., Hron, J., Turner, R. E., & Ghahramani, Z. (2018). Gaussian process behaviour in wide deep neural networks. *arXiv Preprint arXiv:1804.11271*.

</div>

<div id="ref-mcculloch1943logical" class="csl-entry">

McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. *The Bulletin of Mathematical Biophysics*, *5*, 115–133.

</div>

<div id="ref-mnih2013playing" class="csl-entry">

Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. *arXiv Preprint arXiv:1312.5602*.

</div>

<div id="ref-mnih2015human" class="csl-entry">

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, *518*(7540), 529–533.

</div>

<div id="ref-moss2020boss" class="csl-entry">

Moss, H. B., Beck, D., González, J., Leslie, D. S., & Rayson, P. (2020). BOSS: Bayesian optimization over string spaces. *arXiv Preprint arXiv:2010.00979*.

</div>

<div id="ref-neal1995bayesian" class="csl-entry">

Neal, R. M. (1995). *BAYESIAN LEARNING FOR NEURAL NETWORKS* \[PhD thesis\]. University of Toronto.

</div>

<div id="ref-openai2023gpt" class="csl-entry">

OpenAI, R. (2023). GPT-4 technical report. *arXiv*, 2303–08774.

</div>

<div id="ref-opper2000gaussian" class="csl-entry">

Opper, M., & Winther, O. (2000). *Gaussian processes and SVM: Mean field results and leave-one-out*.

</div>

<div id="ref-pearson1901liii" class="csl-entry">

Pearson, K. (1901). LIII. On lines and planes of closest fit to systems of points in space. *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, *2*(11), 559–572.

</div>

<div id="ref-rae2021scaling" class="csl-entry">

Rae, J. W., Borgeaud, S., Cai, T., Millican, K., Hoffmann, J., Song, F., Aslanides, J., Henderson, S., Ring, R., Young, S., et al. (2021). Scaling language models: Methods, analysis & insights from training gopher. *arXiv Preprint arXiv:2112.11446*.

</div>

<div id="ref-ramesh2022hierarchical" class="csl-entry">

Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., & Chen, M. (2022). Hierarchical text-conditional image generation with clip latents. *arXiv Preprint arXiv:2204.06125*, *1*(2), 3.

</div>

<div id="ref-10.7551/mitpress/3206.001.0001" class="csl-entry">

Rasmussen, C. E., & Williams, C. K. I. (2005). *<span class="nocase">Gaussian Processes for Machine Learning</span>*. The MIT Press. <https://doi.org/10.7551/mitpress/3206.001.0001>

</div>

<div id="ref-redmon2016you" class="csl-entry">

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 779–788.

</div>

<div id="ref-rombach2022high" class="csl-entry">

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 10684–10695.

</div>

<div id="ref-romero2013navigating" class="csl-entry">

Romero, P. A., Krause, A., & Arnold, F. H. (2013). Navigating the protein fitness landscape with gaussian processes. *Proceedings of the National Academy of Sciences*, *110*(3), E193–E201.

</div>

<div id="ref-ronneberger2015u" class="csl-entry">

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. *Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18*, 234–241.

</div>

<div id="ref-rosenblatt1958perceptron" class="csl-entry">

Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, *65*(6), 386.

</div>

<div id="ref-roweis1999unifying" class="csl-entry">

Roweis, S., & Ghahramani, Z. (1999). A unifying review of linear gaussian models. *Neural Computation*, *11*(2), 305–345.

</div>

<div id="ref-salimbeni2018orthogonally" class="csl-entry">

Salimbeni, H., Cheng, C.-A., Boots, B., & Deisenroth, M. (2018). Orthogonally decoupled variational Gaussian processes. *Advances in Neural Information Processing Systems*, *31*.

</div>

<div id="ref-seko2015prediction" class="csl-entry">

Seko, A., Togo, A., Hayashi, H., Tsuda, K., Chaput, L., & Tanaka, I. (2015). Prediction of low-thermal-conductivity compounds with first-principles anharmonic lattice-dynamics calculations and bayesian optimization. *Physical Review Letters*, *115*(20), 205901.

</div>

<div id="ref-shahriari2015taking" class="csl-entry">

Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & De Freitas, N. (2015). Taking the human out of the loop: A review of bayesian optimization. *Proceedings of the IEEE*, *104*(1), 148–175.

</div>

<div id="ref-shi2020sparse" class="csl-entry">

Shi, J., Titsias, M., & Mnih, A. (2020). Sparse orthogonal variational inference for Gaussian processes. *International Conference on Artificial Intelligence and Statistics*, 1932–1942.

</div>

<div id="ref-shoeybi2019megatron" class="csl-entry">

Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., & Catanzaro, B. (2019). Megatron-lm: Training multi-billion parameter language models using model parallelism. *arXiv Preprint arXiv:1909.08053*.

</div>

<div id="ref-silver2016mastering" class="csl-entry">

Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam, V., Lanctot, M., et al. (2016). Mastering the game of go with deep neural networks and tree search. *Nature*, *529*(7587), 484–489.

</div>

<div id="ref-snoek2012practical" class="csl-entry">

Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization of machine learning algorithms. *Advances in Neural Information Processing Systems*, *25*, 2951–2959.

</div>

<div id="ref-spearman1904general" class="csl-entry">

Spearman, C. (1904). " general intelligence," objectively determined and measured. *The American Journal of Psychology*, *15*(2), 201–292.

</div>

<div id="ref-srivastava2014dropout" class="csl-entry">

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *The Journal of Machine Learning Research*, *15*(1), 1929–1958.

</div>

<div id="ref-sun2020neural" class="csl-entry">

Sun, S., Shi, J., & Grosse, R. B. (2020). Neural networks as inter-domain inducing points. *Third Symposium on Advances in Approximate Bayesian Inference*.

</div>

<div id="ref-tibshirani1996regression" class="csl-entry">

Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, *58*(1), 267–288.

</div>

<div id="ref-tipping1999probabilistic" class="csl-entry">

Tipping, M. E., & Bishop, C. M. (1999). Probabilistic principal component analysis. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, *61*(3), 611–622.

</div>

<div id="ref-torun2018global" class="csl-entry">

Torun, H. M., Swaminathan, M., Davis, A. K., & Bellaredj, M. L. F. (2018). A global bayesian optimization algorithm and its application to integrated system design. *IEEE Transactions on Very Large Scale Integration (VLSI) Systems*, *26*(4), 792–802.

</div>

<div id="ref-touvron2023llama" class="csl-entry">

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. (2023). Llama 2: Open foundation and fine-tuned chat models. *arXiv Preprint arXiv:2307.09288*.

</div>

<div id="ref-turner2021bayesian" class="csl-entry">

Turner, R., Eriksson, D., McCourt, M., Kiili, J., Laaksonen, E., Xu, Z., & Guyon, I. (2021). Bayesian optimization is superior to random search for machine learning hyperparameter tuning: Analysis of the black-box optimization challenge 2020. *NeurIPS 2020 Competition and Demonstration Track*, 3–26.

</div>

<div id="ref-vaswani2017attention" class="csl-entry">

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, *30*.

</div>

<div id="ref-wigley2016fast" class="csl-entry">

Wigley, P. B., Everitt, P. J., Hengel, A. van den, Bastian, J. W., Sooriyabandara, M. A., McDonald, G. D., Hardman, K. S., Quinlivan, C. D., Manju, P., Kuhn, C. C., et al. (2016). Fast machine-learning online optimization of ultra-cold-atom experiments. *Scientific Reports*, *6*(1), 25890.

</div>

<div id="ref-yang2019machine" class="csl-entry">

Yang, K. K., Wu, Z., & Arnold, F. H. (2019). Machine-learning-guided directed evolution for protein engineering. *Nature Methods*, *16*(8), 687–694.

</div>

</div>
