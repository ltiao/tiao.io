---
draft: true
title: A daily fantasy lineup is a portfolio
summary: We model a locked daily fantasy lineup as a rank-contingent payoff, connect it to portfolio optimization and superhedging, and examine why the apparent arbitrage usually disappears in practice.
date: 2026-08-30
authors:
  - me
categories:
  - technical
---

A traditional daily fantasy sports contest begins with a constrained optimization problem. The operator publishes a *slate*, meaning a set of real sporting events, and assigns a salary to every eligible athlete. You must fill a fixed collection of roster slots without exceeding a salary cap. The resulting roster is your *lineup*.

Pay an entry fee and the lineup joins a *field*: the collection of all lineups entered in that contest. As the real events unfold, a platform-specific scoring rule converts each athlete's statistics into fantasy points. Add the points across the lineup, sort the field by total score, and pay prizes by rank. A simple contest might pay <span>&#36;</span>1,000 to first place, <span>&#36;</span>500 to ranks two through five, <span>&#36;</span>20 to ranks six through fifty, and nothing to everyone else. If several lineups tie across prize positions, they split the affected prizes.

For now, consider a format with one common deadline, called *lock*, after which no lineup can change. Before lock, the lineup is a decision: which athletes should you select? After lock, it is something else. For every possible way the remaining games could unfold, the lineup has a resulting score, a rank against the now-fixed field, and a dollar payout.

In other words, the object that began as a roster has become a function from future states of the world to money.

Now collapse those future states to two. Suppose one event, $E$, is all that remains. If $E$ occurs, our locked lineup finishes high enough to receive <span>&#36;</span>18. If it does not, the lineup receives nothing. We paid <span>&#36;</span>10 to enter, but that cost is already sunk; from this point onward, we are holding a position with the payoff

$$
D(E) =
\begin{cases}
18, & E, \\
0, & \text{not }E.
\end{cases}
$$

Elsewhere, suppose a market sells a binary contract called $\operatorname{NO}(E)$. A binary contract pays a fixed amount if its named event occurs and zero otherwise. This one costs <span>&#36;</span>0.40 now and pays <span>&#36;</span>1 if $E$ does *not* occur. Its payoff is therefore complementary to the lineup's: it pays in precisely the branch where the lineup does not.

A *hedge* is a second position chosen to offset losses in a position we already hold. Buy eighteen $\operatorname{NO}(E)$ contracts for <span>&#36;</span>7.20 and the combined payoff becomes

| Remaining state | DFS payout | $18\times\operatorname{NO}(E)$ payout | Combined payout |
|---|---:|---:|---:|
| $E$ | <span>&#36;</span>18 | <span>&#36;</span>0 | <span>&#36;</span>18 |
| not $E$ | <span>&#36;</span>0 | <span>&#36;</span>18 | <span>&#36;</span>18 |

The DFS entry and hedge cost <span>&#36;</span>10 + <span>&#36;</span>7.20 = <span>&#36;</span>17.20 in total. Either state returns <span>&#36;</span>18. On paper, we have replaced a risky lineup with a guaranteed profit of <span>&#36;</span>0.80.

Did we just manufacture eighty cents, or did we smuggle something into the word *guaranteed*?

## A prize ladder is a bundle of binary claims

Return to the larger contest. Let $R$ denote our lineup's final rank, with $R=1$ best, and let $D(R)$ denote its payout. For the prize ladder above, write

$$
\begin{aligned}
D(R) &= 20\,\mathbf{1}\{R\leq 50\} \\
     &\quad + 480\,\mathbf{1}\{R\leq 5\} \\
     &\quad + 500\,\mathbf{1}\{R\leq 1\}.
\end{aligned}
$$

The indicator $\mathbf{1}\{A\}$ equals one when statement $A$ is true and zero otherwise. A lineup ranked 37th activates only the first term and receives <span>&#36;</span>20. A lineup ranked fourth activates the first two and receives <span>&#36;</span>500. The winner activates all three and receives <span>&#36;</span>1,000. We have reconstructed the entire prize ladder from three yes-or-no questions about rank.

In general, suppose payouts $P_1>P_2>\cdots>P_K>0$ apply at rank cutoffs $r_1\lt r_2\lt\cdots\lt r_K$, and set $P_{K+1}=0$. Then

$$
\begin{aligned}
D(R) &= \sum_{k=1}^{K}(P_k-P_{k+1}) \\
     &\qquad \cdot \mathbf{1}\{R\leq r_k\}.
\end{aligned}
$$

Each summand is a *binary rank claim*: an all-or-nothing layer of prize money triggered by crossing one rank cutoff. The platform does not sell these layers separately. The identity says only that its tiered payout behaves *as if* the lineup contained a bundle of them.

This is the result I initially missed in Haugh and Singal's *How to Play Fantasy Sports Strategically (and Win)*.[^haugh-singal] Their equation (2) takes the expectation of this same decomposition: each indicator becomes the probability of beating the relevant rank cutoff, and those probabilities enter with the successive prize differences. They also note the tie correction. When a tied group straddles a cutoff, the corresponding indicator is replaced by the fraction of that prize layer the lineup receives.

I had searched the paper for an external hedge, correctly found none, and incorrectly reported that the underlying connection was absent. Haugh and Singal use the decomposition to choose lineups before lock; they do not price its binary pieces, match them to sportsbook contracts, or trade after the field is fixed. The rank-claim representation was nevertheless already there, in the objective function.

So the decomposition was not the new part. The remaining question was whether its virtual binary pieces could be matched by real binary contracts somewhere else.

[^haugh-singal]: Martin B. Haugh and Raghav Singal, [“How to Play Fantasy Sports Strategically (and Win)”](https://doi.org/10.1287/mnsc.2019.3528), *Management Science* 67(1), 2021.
