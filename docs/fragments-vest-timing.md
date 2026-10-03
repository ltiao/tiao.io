# When to sell your RSUs: a pre-registered answer

Fifty-six earnings quarters, twelve strategies, one frozen decision rule, and
the answer is the one your index-fund friend would have given you for free:
sell on vest day. The difference is that now it's *my* answer — I know what it
survived.

---

The result was pre-registered as a success before it existed: "sell on vest
day, confirmed" ends the quarterly deliberation permanently. A null result
that buys back four decisions a year.

---

The zero block. The conditional arms — the ones the whole project existed to
test — score excess of exactly 0.0 whenever their gate stays shut, because
"don't fire" means "do what the baseline does." C1 fires on 31 of 56 events
and averages +0.24% when it acts, but neither its wins nor its losses reach
half the sample, so the median rank lands inside the block of zeros and the
median is pinned at exactly 0.0. Not slightly negative. Exactly zero. The
rule we froze was structurally blind to the arms we most wanted to test — and
we only noticed after the verdict.

---

Pre-registration has a resolution limit. We froze twelve arms, five
constants, a bootstrap seed, and a decision rule — and the pre-tax verdict
still hinged on a parameter nobody thought to name: the quantile
interpolation convention. Under two of pandas' five methods, hold-8-quarters
qualifies. Under the other three, nothing does. The fix was disclosure, not
adjustment: the convention was inherited blind before any result existed, so
it stays — but the sensitivity goes in the report, because "nothing was tuned
after seeing the numbers" was true and incomplete.

---

The half-split was the executioner. Hold-4-quarters and hold-8-quarters look
good on the full 2012–2026 sample — of course they do; META rose roughly
fifteenfold from its IPO over that stretch (28-fold from the first vest). Split
the events in half by date and the edge lives entirely in the first half. An
arm that only wins in one era is noise wearing a trend.

---

The tests were built to catch specific lies. The DCA fixture gives the index
fund a different price on each of the three sale dates, so that an
implementation that quietly parks all the proceeds from the last sale date
produces 149.99999999999997 instead of 130.0 — and fails. The 52-week-high
test hides a higher price one trading day outside the lookback window, so an
off-by-one can't sneak past. I stopped asking "does the test pass?" and
started asking "which wrong program would this test catch?" — a question that
changed how I write tests for everything, not just backtests. (First-person,
no tooling visible.)

---

VOICE DIRECTIVE (author's explicit instruction, overrides the usual
posts-are-deadpan default): no dry register — earnest wonder in the style of
John Green. Direct address, sincerity as a stance, digressions that pay off,
self-deprecating honesty about one's own temptations, wonder located in
ordinary machinery (a trading calendar, a median). Take a small thing
seriously enough and it opens onto everything. No star-ratings or other
Anthropocene Reviewed surface mimicry — the sincerity, not the format.

---

SPINE DIRECTIVE: this is the finance story, first-person singular — my
shares, my quarterly ritual, my machine for not fooling myself. The article
must NOT look agent-driven at all: no subagents, no reviewers, no
orchestration, no AI tooling anywhere in the text. Verification appears only
as things *I* did — "I re-derived every expected value by hand," "I wrote the
test before the code." Where existing fragments say "we," the article says
"I."

---

STRUCTURE DIRECTIVE: anchor the whole piece on multiple beautiful, crisp
visualizations — the kind that make a reader go "huh, neat" or "I hadn't
thought of visualizing it from that perspective." Tell the story in one
breath: the figure sequence IS the narrative arc; prose connects and
propels, it doesn't lecture. Each figure earns its place by carrying a beat
no paragraph could.

---

Candidate visualizations (the figure sequence is the argument):

- **The clockwork.** One year drawn as the quarterly machine every Meta
  employee lives inside but has never seen drawn: earnings call (Wednesday,
  after close) → window opens (Friday) → shares land (the 15th, mid-window)
  → window closes ~30 days later. Repeat ×4. A "you are here" marker on vest
  day. Confession beat, not reader-surprise: *I* had assumed the decision
  started when the window opened. It doesn't — the shares land mid-window,
  on the 15th. My mental model of my own compensation was off by two weeks,
  and I only found out when I had to draw the machine I live inside. (Do not
  project the error onto the reader; own it.)
- **The cloud.** All 56 events overlaid: META price path normalized to 1.0
  at vest day, faint spaghetti, median path bold. The honest answer to
  "should I wait a few days?" is not a line, it's this fan. Wonder beat:
  the sheer width of it.
- **The zero block.** Per-event excess for C1, sorted: losses on the left,
  a wall of *exact* zeros in the middle, wins on the right — and the median
  rank landing inside the wall. The picture that explains why a strategy
  that makes money when it acts can still have a median of exactly 0.0.
  This is the centerpiece "hadn't thought of it that way" figure.
- **The executioner.** Hold-8-quarters' per-event excess plotted by vest
  date, first half of history vs second half shaded as eras. The full-sample
  edge visibly living only in the first era. Noise wearing a trend, drawn.
- **The footnote that almost flipped the verdict.** Five quantile
  conventions, one threshold line, hold-8q's statistic hopping across it
  under two of the five. The verdict as a function of a parameter nobody
  names.
- **The cliff.** After-tax proceeds vs holding period: the LTCG
  discontinuity at one year and a day. The only place the tax code draws in
  this problem, and it's a literal step function.
- **The ribbons.** Every arm's P5–P95 excess ribbon vs baseline at each
  horizon, all straddling zero. Twelve strategies, one gray smear across
  the axis. The null result as a single picture.

---

PRODUCTION DIRECTIVE: figures are static, one consistent style, built as
standalone TikZ/pgfplots documents and exported to both SVG and 300-dpi PNG
by script from the project's own cached data (pipeline + outputs live in the
vest-timing repo under figures/) — every figure reproducible with
`uv run python figures/prep_data.py && figures/build.sh`. No interactivity
anywhere; the stillness is part of the argument. THE FIGURES ARE ALREADY
BUILT — the article references them from figures/out/, it does not recreate
them.

---

Selling on vest day is the only free move on the board. Your cost basis is
set at the vest-day price, so selling that day realizes a gain of zero —
no tax event worth the name. Every day you wait, you're building a position
in a single stock with someone else's conviction and your own tax bill.
Holding isn't "not deciding." Holding is buying META with a cash bonus you
were never handed.

---

Honesty about the sample, stated with affection rather than apology: this is
one stock, one historical path, fifty-six events, and the long-horizon
windows overlap so heavily that the effective sample is smaller than it
looks. No bootstrap can manufacture history that didn't happen. The backtest
can't tell you what META will do; it can tell you whether *any* of the
stories people tell at lunch survived contact with the only history we have.
None did.

---

The decision rule existed because I know myself. Without teeth decided in
advance, I would have stared at a suggestive mean excess and talked myself
into it — everyone does; that's what a suggestive mean excess is *for*. So
the rule was frozen before any number existed: beat the baseline on median,
keep the 25th percentile above catastrophe, survive both halves of history,
at every horizon, or the answer is the boring one. The rule's job was never
to find a winner. Its job was to protect me from finding one.

---

Why I thought timing might work at all (the setup beat): post-earnings
announcement drift is one of the most documented anomalies in finance —
prices keep drifting in the direction of an earnings surprise for weeks.
My shares land two weeks after earnings, mid-drift, every single quarter, by
construction. If PEAD was ever going to pay a retail human, it would pay
*here*, on schedule, four times a year. That's the trap door the whole
project walks over: the setup is perfect, and the answer is still no.

---

The emotional beat (author's own, confirmed): vindication. I suspected the
boring answer all along; I built the machine to earn the right to believe
it. But write it with the self-awareness that makes vindication honest:
wanting the boring answer to win is a bias too. If you secretly root for
the null, you might unconsciously build a courtroom where your favorite
verdict always wins — which is exactly why the rule had to be frozen before
any number existed. The pre-registration protected the result from my
cleverness *and* from my humility.

---

The ending that opens outward (conditional-on-firing): the null result
produced a better question than the one I started with. The median criterion
judged every arm on all fifty-six events, including the ones where a
conditional arm chose to do nothing — and doing nothing scores exactly zero,
so the arms I most wanted to test were graded mostly on their silence. The
natural next read is conditional-on-firing: how does C1 do *on the events
where it actually acted*? I can't promote that to a real test — I thought of
it after seeing the verdict, which makes it exploratory forever, a hypothesis
for someone else's pre-registration (or my next one). But that's the quiet
gift of a frozen rule: even its blind spots are informative, and they tell
you exactly what the next experiment should be.

---

Disclaimer, in-voice, one breath, no boilerplate (candidate sentence):
"None of this is financial advice. It's one person's attempt not to fool
himself, with the work shown. My stock, my taxes, my history — the method
transfers; the verdict might not." Place it where it reads as honesty
rather than legal reflex — near the verdict, not in a footer. (Site also
has a standing /disclaimer page; link it quietly if the theme allows.)
