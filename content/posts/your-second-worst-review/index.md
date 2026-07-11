---
draft: true
title: Your second-worst review is probably all you need
summary: What nine years of ICLR reviews say about which score carries the most weight in your paper's fate.
date: 2026-07-10
authors:
  - me
tags:
  - Peer Review
  - Machine Learning
  - Data Visualization
---

Two low scores on your ICLR paper is a death sentence. One savage review, on its own, you can survive. Two, and it is over: about a one-in-fifty shot, however enthusiastic your other reviewers are, and no rebuttal claws it back. The score that matters most is not your worst review. It is your *second*-worst. I know this because ICLR publishes its rejections, and nine years of them tell the same story.

That last part does more work than it looks. All three of the big machine-learning conferences run on OpenReview, so you might assume all three make their reviews public. They do not. NeurIPS and ICML post the papers that got in and quietly drop the ones that did not, reviews and all. Scrape NeurIPS off OpenReview today and it looks like the conference accepts 95% of what it receives, because the only rejections left standing are the handful of authors who chose to un-hide theirs. ICLR is the exception. It leaves the rejected papers up, scores and reviews intact, thousands of them a year going back to 2018. So when I say your second-worst review decides your fate, I can only say it about ICLR, because ICLR is the only one of the three that will show you the papers that lost.

Here is the part that convinced me it was real. Over those nine years, ICLR could not make up its mind about how to score a paper. Some years reviewers rated on a full 1-to-10 slider (2018, 2019, 2021). Other years it was a coarse ladder of a few allowed values. In 2020 those were 1, 3, 6, and 8. From 2022 they were 1, 3, 5, 6, 8, and 10, with a deliberate hair-splitting step between a score meaning "marginally below the bar" and one meaning "marginally above" it. In 2026 the scale jumped onto even numbers, 0 through 10. Four different score sheets in nine years. The second-worst-review effect turns up, unmistakable, on every one of them. Whatever it measures, it is not an artifact of the rubric, because the rubric kept changing and the effect never did.

So here it is. Every ICLR 2024 paper that drew four reviewers, arranged by its two lowest scores (which panel it falls in) and its two highest (which square inside the panel), each square shaded by how often papers like it got in.

![Grid of small heatmaps showing ICLR 2024 acceptance rate broken down by a paper's two lowest and two highest reviewer scores. A block of near-zero, deep-red acceptance fills the top-left panel, where both of the two lowest scores are 3; the rest of the grid runs green.](coalition-heatmap.png)

Your eye finds the red before you read a single label. That corner is the papers whose two lowest scores both landed on 3, and it is a wall: nothing in the squares to its right climbs out of it. The whole finding is the shape of that boundary.

If you are a researcher with a regression reflex, your first objection is probably: isn't the mean score a better predictor? It is. A logistic regression on the mean alone gives a cross-validated AUC of 0.94; the second-lowest score alone gives 0.89. The mean wins because it uses all four scores at once, which is strictly more information. But the question is not which summary statistic predicts best. It is which of the four underlying scores the decision actually hinges on.

The four headline scores together predict acceptance at an AUC of 0.95 in held-out folds. I tried adding 44 more features (sub-scores for soundness, presentation, and contribution, reviewer confidence, word count, question count, submission timing) and could not improve on that, though with roughly 3,700 papers and correlated predictors I would not read too much into the absence of a gain.

Within those four scores, every method I tried ranked the second-lowest first. A decision tree, forced to pick one variable to split on at the root, picks s2 at the 5/6 boundary. Logistic regression gives it the largest standardized coefficient, roughly double the minimum score and triple the maximum. Random forest and gradient boosting permutation importance agree. Shapley values on a gradient-boosted model give s2 a mean absolute contribution of 2.06, versus 1.17 for the max, 1.13 for the third score, and 0.61 for the min. For 58% of individual papers, s2 is the single score that swings the prediction the most. For another 25% it is the third score, and for 16% the highest. Your worst review, the one you are staring at, is the most important score for fewer than 2% of papers. All four scores matter, but s2 is first among them, and s1 is last by a wide margin.

![SHAP beeswarm plot showing each paper as a dot, colored by score value, spread along the x-axis by how much that score pushed the acceptance prediction. The 2nd-lowest row has the widest spread; the lowest row is a tight cluster near zero.](shap-beeswarm.png)

The reason is what it takes to look like noise. One low score is a dissent. Area chairs see dissents all the time and know how to dismiss them: a reviewer who skimmed too fast, a mismatch in taste, an off day. A second low score is no longer a dissent. It is two independent readers arriving at the same conclusion, and that is much harder for an area chair to wave away. The veto, in other words, needs a second. Your worst review is not what dooms you because a single outlier can be explained; your second-worst is what dooms you because once there are two, they stop looking like outliers.

To see how stubborn the wall is, look at the best-case rescue: a paper with two 3s and two 8s. The highest top pair the scale allows, and the lowest floor. In 2024, those papers got in about 2% of the time, against a baseline of roughly 40%. In 2026, on the new scale, two 2s with two 8s fared a little better at 25%, still well under the conference average. Two strong champions cannot buy off two detractors. The floor holds.

So the next time your scores come in, do not stare at the lowest one. Look at the one above it. That is the number that probably already sealed its fate.
