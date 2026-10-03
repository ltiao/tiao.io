---
title: A prediction-market edge that was really the easing cycle
subtitle: ""
summary: We backtest a trade that bets Kalshi's Fed-decision markets against CME FedWatch's futures-implied probabilities. It clears a pre-registered bar — and we show the apparent edge is beta to the 2024–25 rate-cutting cycle, with FedWatch adding nothing over Kalshi's own price.
date: "2026-07-07T00:00:00Z"
draft: true
featured: false
categories:
  - technical
authors:
  - me
tags:
  - Prediction Markets
  - AI Agents
  - Backtesting
image:
  focal_point: Smart
  preview_only: false
projects: []
---

Eight times a year, the U.S. Federal Reserve decides whether to raise interest
rates, cut them, or leave them alone. You can bet on that decision in two very
different places. CME FedWatch reads a probability for each outcome out of
interest-rate futures — the professional market where that risk is hedged — and
publishes it: a 71% chance of no change, a 22% chance of a quarter-point cut, and
so on. Kalshi is a regulated exchange where anyone can put a few dollars on the
same outcomes. When the two disagree, there is an obvious trade: buy the outcome
Kalshi has priced too cheaply against the futures, and hold it until the meeting
settles the question.

I backtested that trade across 26 meetings. It passes every test I put in front
of it — it makes money in each of the held-out validation years, and the
confidence interval sits clear of zero. And it is still not a trade. What looks
like an edge is the Fed's 2024–25 rate-cutting streak wearing a disguise.

The premise is that these two prices are set by different crowds. Fed funds
futures are traded by banks and funds with real money riding on interest rates;
the probability FedWatch extracts from them is about as sharp a forecast of the
Fed as exists. Kalshi's Fed markets are mostly retail — people betting twenty
dollars on a hunch. If the retail crowd is even slightly worse at pricing the
Fed, then FedWatch is the better estimate, and any gap between the two is Kalshi's
mistake to collect.

But that story hides an assumption: it only works if the Kalshi crowd isn't
already looking at the futures. A reference price earns you nothing when the
market is anchored to the same information — you are just two thermometers reading
the same room. So the real question is not whether FedWatch is a good forecast (it
is), but whether Kalshi already knows what FedWatch knows.

{{< figure src="two-prices-one-meeting.png" alt="Grouped bar chart of FedWatch versus Kalshi probabilities for the five FOMC outcomes of a single meeting" caption="The two prices, side by side. FedWatch's futures-implied probabilities and Kalshi's retail prices for the five possible outcomes of a single FOMC meeting, about a month out. They agree on the favorite; Kalshi just prices it a little cheaper, and that gap is the trade." >}}
<!-- FIGURE-TODO: grouped bar chart, categories {Cut >25bp, Cut 25bp, Hold, Hike 25bp, Hike >25bp}, two bars per category (FedWatch vs Kalshi), y = probability 0-1. Use a meeting where both agree on the favorite but Kalshi is cheaper (e.g. 2024-12-18 or 2025-09-17), from the backtest panel at the entry date. -->

The analysis was run largely by AI agents — Claude Opus 4.8, driven through Claude
Code — under my direction. They did more than write and run the backtests: I
orchestrated them in dynamic multi-agent workflows, where a single script fans out
a dozen or more independent agents, each writing its own code against the same raw
data, then has them cross-check and adversarially attack one another's
conclusions. Every result below survived a fourteen-agent workflow of exactly that
kind — which is why I trust it more than a single script written once and never
checked.

The data comes from two places. FedWatch's numbers are CME's own end-of-day
output, pulled from its data feed — a paid entitlement, not a scrape — and parsed
directly rather than reconstructed, so they are exactly the probabilities CME
published, not my approximation of them. The feed is *point-in-time*: every daily
reading is stamped with the date it was released, so the backtest sees only what
was knowable on each day and nothing that leaked in from later. The Kalshi side is
the exchange's public market data — for each Fed-decision contract, a daily candle
with the closing price, the day's best bid and ask, and, once the meeting is over,
the settled result: which outcome actually won. FedWatch's history runs back years,
but its overlap with a liquid Kalshi contract is what bounds the study to 26
meetings, mid-2023 to mid-2026.

Matching the two is the fiddly part. Each Kalshi contract is one outcome — "one
25 bp cut," "no change," and so on — and it settles on the Fed's move *at that
meeting*, relative to the rate in effect going in. FedWatch reports the probability
of each rate level, so lining the two up means measuring every FedWatch reading
against that same going-in rate. Get the reference point wrong and probability from
one meeting bleeds into the next; it is the single easiest thing in the whole
pipeline to get quietly wrong.

The rest of the backtest is deliberately plain. A bet is entered the first day the
gap between the two prices crosses a threshold, filled at Kalshi's asking price with
its trading fee subtracted — so the backtest pays to cross the spread rather than
assuming a free fill. The 26 meetings split into training (through 2023), validation
(2024–25), and a test year (2026) held in reserve; every decision is made on
validation, and the test year is looked at once, at the very end.

One number frames everything that follows: the Fed meets eight times a year, and
only one or two outcomes per meeting are ever a cheap-enough favorite to bet. The
whole strategy fires about twenty times across three years. This is a small-data
problem, and small data is where false edges are born.

| Fold | Years | Meetings | Bets |
|------|-------|---------:|-----:|
| Train | ≤ 2023 | 6 | 5 |
| Validation | 2024–25 | 16 | 12 |
| Test | 2026 (partial) | 4 | 3 |

*The entire dataset. FedWatch end-of-day probabilities exist for 26 meetings; the
strategy places 20 bets, most of them in the two validation years.*

{{< figure src="meeting-timeline.png" alt="Timeline of FOMC meetings 2023 to 2026 colored by train, validation, and test fold" caption="Every FOMC meeting in the sample on a three-year timeline, colored by fold. Filled markers are meetings where the strategy actually placed a bet — eight meetings a year, a handful of bets, and that is the whole sample." >}}
<!-- FIGURE-TODO: horizontal strip/timeline plot, x = date (2023-06 -> 2026-07), one marker per meeting; color = fold (train/validation/test); filled vs open marker = bet vs no bet. -->

## It clears the bar

With twenty bets you can talk yourself into almost anything, so the honest move is
to fix the test before you look. I set two gates a skeptic would insist on: the
trade has to make money across the validation years after fees, with a confidence
interval clear of zero, and the sign has to hold in *each* validation year on its
own — not rescued by a single lucky one.

Done carefully, with that reference-matching handled, the trade clears both. The
average bet returns about 23 cents on the dollar risked. 2024 and
2025 are each positive on their own, the held-out 2026 year is positive too, and
the confidence interval on the validation return sits above zero. By the numbers,
this looks like a real edge.

| Fold / year | Bets | Mean return per \$1 risked |
|-------------|-----:|---------------------------:|
| **Validation (2024–25)** | 12 | **+0.23** |
| &nbsp;&nbsp;2024 alone | 4 | +0.05 |
| &nbsp;&nbsp;2025 alone | 8 | +0.32 |
| **Test (2026)** | 3 | +0.36 |

*Corrected-reference results. Positive in every held-out year; the validation 90%
confidence interval clears zero. By these numbers it is a real edge — which is the
whole problem.*

{{< figure src="validation-return-bootstrap.png" alt="Histogram of bootstrapped validation mean returns with the 90 percent interval above zero" caption="Bootstrap distribution of the validation mean return, resampling whole meetings (bets in the same meeting rise and fall together, so the meeting is the unit). The 90% interval sits above zero: the trade clears its pre-registered statistical gate." >}}
<!-- FIGURE-TODO: histogram of ~10k bootstrapped mean per-bet returns (resample by meeting); vertical rules at the 5th and 95th percentiles and at 0; annotate the point estimate +0.23. -->

## Why it isn't a trade

Passing those two gates is necessary, not sufficient. A real edge has to survive
three more questions, and this one fails all three.

*Does FedWatch actually disagree with Kalshi about what will happen?* Hardly.
Across the 26 meetings, FedWatch and Kalshi name the same most-likely outcome in 24
of them. They quibble over the price of the favorite; they almost never disagree
about which outcome it is. The "retail underprices what the futures know" story
needs the two crowds to see the world differently, and mostly they don't.

{{< figure src="favorite-agreement.png" alt="Scatter of Kalshi price versus FedWatch probability for each meeting's favorite outcome, points hugging the diagonal" caption="Each point is a meeting: the horizontal axis is Kalshi's price for its own favorite outcome, the vertical axis is FedWatch's probability for that same outcome. They hug the diagonal — the two name the same favorite in 24 of 26 meetings and differ only on its price." >}}
<!-- FIGURE-TODO: scatter plot, both axes probability 0-1, with a y = x reference line; highlight the 2 meetings where the favorites disagree. -->

*Does betting only on the disagreement beat ignoring it?* No. Drop the rule that
says "only bet when the gap is wide enough" and just buy every FedWatch favorite —
you make the same money. The gate that was supposed to isolate Kalshi's mistakes
selects nothing; the disagreement carries no reliable information over simply
buying the favorite.

*Where does the money come from?* One place. Split the bets by which outcome they
backed, and every dollar of the edge is in "one quarter-point cut" bets — placed in
2024 and 2025, when the Fed cut at nearly every meeting. The "no change" bets, over
the very same window, lose money. You did not find a pricing error. You bought rate
cuts during a rate-cutting cycle, and the cuts arrived.

| What you bet on | Bets | Mean return / \$1 |
|-----------------|-----:|------------------:|
| The full trade (only on disagreement) | 12 | +0.23 |
| Buy every FedWatch favorite instead | 19 | +0.18 |
| The trade's "one 25 bp cut" bets | 6 | **+0.48** |
| The trade's "no change" bets | 6 | **−0.02** |

*Rows 1–2: gating on the disagreement doesn't beat ignoring it — the gap is within
noise. Rows 3–4: the trade's own bets, split by outcome, all the profit is rate
cuts and the "holds" lose.*

{{< figure src="return-by-outcome.png" alt="Bar chart of mean return per bet grouped by the outcome backed, showing cuts positive and holds negative" caption="Mean return per bet, grouped by the outcome the bet backed. The edge is entirely 'one 25 bp cut' bets during the 2024–25 easing cycle; 'no change' bets are negative." >}}
<!-- FIGURE-TODO: bar chart, x = outcome bucket (Cut 25bp, Hold, others), y = mean return per $1, zero line marked. -->

## Two equally good forecasts

Step back to the premise: FedWatch was supposed to be the sharper forecast, the
smart money the retail crowd hadn't caught up to. That is a claim you can test
directly, with no trade at all — take every probability each side published,
compare it to what the Fed actually did, and score the two as forecasters.

The standard score is the Brier score: roughly, the mean squared error between "you
said 80%" and "it happened (1) or it didn't (0)," lower being better. Scored this
way, at every horizon from two months out to the eve of the meeting, FedWatch and
Kalshi's own price are a statistical tie — neither reliably the better forecaster.
This is the one place it's easy to fool yourself: measure only the day before the
meeting, when both have converged, and you find nothing either way. The comparison
has to be made where the trade enters, a month or more out — and even there, they
tie.

{{< figure src="calibration-reliability.png" alt="Reliability diagram comparing FedWatch and Kalshi calibration curves, both near the diagonal" caption="Do the forecasts mean what they say? Predicted probability versus realized frequency, binned, for FedWatch and for Kalshi's price. Both track the diagonal about equally — neither is the sharper forecaster." >}}
<!-- FIGURE-TODO: reliability diagram, x = predicted probability (binned 0-1), y = observed frequency, two series (FedWatch, Kalshi), y=x reference line, marker size proportional to bin count. -->

{{< figure src="brier-by-horizon.png" alt="Brier score versus days to meeting for FedWatch and Kalshi, two overlapping lines with confidence bands" caption="The same verdict at every horizon. Brier score (lower is better) for FedWatch versus Kalshi's price against days to the meeting; the two overlap within noise from two months out to the eve. The trade enters at the left, and even there they tie." >}}
<!-- FIGURE-TODO: line plot, x = days-to-meeting (~60 -> 0), y = Brier score, two lines (FedWatch, Kalshi) with bootstrap CI bands. -->

Which returns us to the two thermometers. FedWatch is a genuinely good forecast of
the Fed. It is also worth nothing as a signal against Kalshi, because Kalshi is an
equally good forecast, as far as two years of data can tell. The retail crowd is
not ignoring the futures; it has priced them in. There was never a gap to collect —
only the coincidence that, for two years, the cheap favorites happened to be rate
cuts and the Fed happened to be cutting.

## No gap to collect

So there is no trade here — or at least not this one. What looked like an edge was
the easing cycle in disguise, and the disguise was good enough to clear a
pre-registered bar. That is the real lesson: a backtest that makes money every year
with a confidence interval off zero is not evidence of an edge. It is the start of
the interrogation, not the end.

The general version is worth keeping. A reference price only pays when the market is
anchored to something dumber than your reference. Weather markets fail this because
the crowd already reads the same forecasts you would; the Fed fails it because
Kalshi already reads the same futures. The question to ask of any "my reference
beats the crowd" idea is not whether the reference is good, but whether the crowd is
already using it — and that is answerable, cheaply, before you write a line of
trading code.

{{< figure src="reference-vs-market-quadrant.png" alt="Two-by-two diagram of reference quality versus whether the market already uses it, with one profitable quadrant highlighted" caption="When a reference beats the market. A reference price only pays in one corner — where it is a good forecast and the crowd isn't already using it. Weather markets and the Fed are both good references that fail on the second axis." >}}
<!-- FIGURE-TODO: 2x2 conceptual schematic (no data). x-axis = 'is the reference a good forecast?' (no -> yes); y-axis = 'is the market already using it?' (yes -> no). Highlight the (good forecast, market NOT using it) quadrant as the only profitable one; place 'Weather' and 'Fed' in the (good, already-used) quadrant. -->

I haven't buried this one. A pattern that only appears during an easing cycle
deserves one honest test: watch it forward, live, into a stretch where the Fed isn't
cutting, and see whether anything survives. My bet is that nothing does — but that
is a prediction, not a result, and the distance between the two is the whole point.

