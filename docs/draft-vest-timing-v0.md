# A Machine for Not Fooling Myself About My Own Stock

Four times a year a batch of Meta shares lands in my brokerage account, and
four times a year I have the same argument with myself about what to do with
them. One side of the argument is boring. It says sell them today, move the
proceeds into a broad index fund, and go do something else with the afternoon.
The other side is far more interesting. It says wait. Just a few days. Just
until the drift finishes. Just until the window closes. The interesting side
has never once produced evidence, and it has won more of these arguments than I
would like to tell you.

So I decided to settle it the only way I trust, which is to write down in
advance exactly what would have to be true for me to change my behavior, and
then go check every quarter since 2012 to see whether it ever was.

The first thing I learned had nothing to do with returns. To simulate a
decision you have to know exactly when the decision exists, which meant drawing
the calendar I have lived inside for years without ever once looking straight at
it.

![The quarterly clockwork](figures/out/fig1-clockwork.png)

*One year of the machine, with a single cycle enlarged beneath it.*

Here is my confession. I had always assumed the decision began when the trading
window opened. It does not. The window opens a couple of days after the
earnings call, and the shares themselves arrive in the middle of it, more than
two weeks later. Everything I believed about the shape of my own choice was off
by roughly a fortnight, and I found out only because I had to draw the machine
in order to simulate it. You can live in an apartment for a decade without ever
learning which wall holds up the roof. I had been living inside this one since
2012.

There was a real reason to think waiting might pay, and it is a better reason
than the ones that get offered over lunch. Post-earnings announcement drift is
one of the most stubbornly documented anomalies in finance: after a company
surprises the market, its price tends to keep drifting in the direction of that
surprise for weeks afterward, and nobody has managed to explain it away in
decades of trying. Now look at where my shares land. Two and a half weeks after
the earnings call, in the middle of the drift, every single quarter, by
construction rather than by any cleverness on my part.

That is the trap door this whole project walks over. If post-earnings drift
were ever going to pay an ordinary person with a brokerage account and no
special information, it would pay here, on a schedule, four times a year, for
as long as I keep showing up to work.

Before testing a single strategy I plotted the raw material: every vest since
2012, each price path rescaled so that vest day equals one.

![The cloud](figures/out/fig2-cloud.png)

*Fifty-six vests, each normalized to its own vest-day price, with the median
path in bold.*

I sat with this one for a while. The median climbs, because META climbed, and
that part is just the history we happened to get. But look at the right-hand
edge. Same company, same holding period, same everything except which quarter
you happened to be born into as a shareholder, and the gap between the luckiest
vest and the unluckiest one is the gap between nearly tripling your money and
losing most of it. Any edge I was hoping to find would have to be a faint signal
living inside all of that. I think this picture belongs on the inside cover of
every brokerage app.

Which is exactly why I wrote the decision rule before I looked at anything else.
I know myself. Hand me a suggestive average at nine in the evening with nothing
frozen in advance, and I will talk myself into it, and so would you, because
talking people into things is what a suggestive average is for. So the teeth
were set first. To beat selling on vest day, a strategy had to win on the
median, keep its 25th percentile above catastrophe, and do both in each half of
history, at every horizon. Otherwise the answer stays boring. The rule's job was
never to find me a winner. Its job was to stop me from finding one.

I also decided in advance that the boring answer would count as a success. This
matters more than it sounds. If "sell on vest day, confirmed" is a real
outcome rather than a failure to find anything, then a null result ends the
quarterly deliberation permanently and buys back four decisions a year, every
year, for the rest of my working life.

Twelve strategies went into the grid, counting the baseline of selling on vest
day and moving the money into a total-market fund. The challengers: sell two,
five, or ten trading days later; spread the sale across three dates; hold for
one, two, four, or eight quarters; and three conditional arms that behave
exactly like the baseline unless their own signal fires, keyed to a large
earnings reaction, a gated version of the same, and a 52-week high. Fifty-six
vests, measured at one, two, and four years. The longer horizons score fewer
events, because history runs out before the holding period does.

I wrote the tests before the code, and somewhere in the middle of that the
question I was asking quietly changed. I stopped asking "does this test pass?"
and started asking "which wrong program would this test catch?" Those sound
like the same question and they are not. The dollar-cost-averaging fixture, for
example, gives the index fund a deliberately different price on each of the
three sale dates, so that an implementation which quietly parks all the
proceeds at the final price returns 149.99999999999997 instead of 130.0 and
fails loudly. The 52-week-high test hides a higher price one trading day
outside the lookback window, so that an off-by-one cannot slip past wearing a
green checkmark. That reframing has since changed how I write tests for
everything, which is a strange thing to get out of a backtest about your own
paycheck.

The arm I actually cared about was the earnings-reaction one, the direct heir
of the drift hypothesis: when the market's reaction to earnings is large, wait;
otherwise do what the baseline does. It fires on 31 of the 56 vests, and when
it fires it makes money on average, a bit under a quarter of a percent per
event at the one-year horizon. The decision rule could not see any of that, for
a reason that took me an embarrassingly long time to understand and about four
seconds once I finally drew it.

![The zero block](figures/out/fig3-zeroblock.png)

*Every scored event for the earnings-reaction arm, sorted worst to best.*

It is one thing to know abstractly that a median is a rank statistic, and quite
another to watch your favorite hypothesis get graded, mostly, on its silence.
The rule I had frozen was structurally blind to the arms the entire project
existed to test, and I did not notice until after the verdict was in. There is a
particular feeling attached to that: the machine running exactly as specified,
doing precisely what I told it to do, and declining to answer my favorite
question.

Meanwhile the arms I did not care about were behaving suspiciously well.
Holding for four or eight quarters looked terrific across the full 2012 to 2026
sample. Of course they did. META rose roughly fifteenfold over that span, so any
rule amounting to "own more META for longer" was always going to look like
genius. This is precisely why the frozen rule demanded both halves of history.

![The executioner](figures/out/fig4-executioner.png)

*Every four-year outcome for the hold-eight-quarters arm, plotted by the date
its shares vested.*

The dots walk downhill as the sample gets younger, and the full-sample edge
turns out to live entirely in the years before 2017. An edge that exists only
in the first half of the record is not an edge. It is the shape of one
particular decade, and I would be paying for it with the next one. Had I
computed the full-sample number and stopped there, I would have adopted this
strategy and felt clever about it for years.

And then, with the verdict looking clean, it wobbled, on a parameter that
nobody involved had thought to freeze. Ask twenty numbers for their median and
there is no unique middle, so somebody has to decide what to do between the
tenth and the eleventh. There are five standard answers. I had inherited one of
them from a library default.

![The footnote that almost flipped it](figures/out/fig5-footnote.png)

*The statistic the pre-tax verdict hangs on, computed five different ways.*

I had frozen a dozen strategies, five constants, a random seed, and a decision
rule with teeth, and the verdict still turned on an argument default I had
never read. The honest fix was disclosure rather than adjustment. The
convention was chosen blind, before any comparative result existed, so it
stays, and the sensitivity goes into the report, because "nothing was tuned
after seeing the numbers" was true and incomplete. Pre-registration has a
resolution limit. You can only freeze the parameters you know exist, and you
find out about the others exactly the way I did.

Here, then, is everything at once.

![The ribbons](figures/out/fig7-ribbons.png)

*All eleven challengers against the baseline, at each of the three horizons.*

Not one arm qualifies. Not at any horizon, not before tax, and not under an
illustrative tax model either. Twelve strategies, three horizons, one frozen
rule, and the adopted policy is the one my index-fund friend would have given
me for free over coffee: sell on vest day. The difference is that it is now
*my* answer, because I know exactly what it survived. None of this is financial
advice. It is one person's attempt not to fool himself, with the work shown. My
stock, my taxes, my history. The method transfers; the verdict might not. (The
site keeps a standing [disclaimer](/disclaimer) if you want the version written
by someone more careful than me.)

The part I find genuinely lovely is what the tax code does to all this, which
is almost nothing, and then one thing.

![The cliff](figures/out/fig6-cliff.png)

*The tax code's entire contribution to this decision, which is one step and
nothing else.*

Selling on vest day is the only free move on the board. The cost basis is set
at the vest-day price, so a sale that day realizes a gain of essentially zero.
The shares were already taxed as income the moment they landed, and the sale
adds nothing to that. Every day you hold afterward, you are building a
concentrated position in one company on somebody else's conviction, with a tax
bill quietly accruing in your name. Holding is not the absence of a decision.
Holding is buying META with a cash bonus you were never handed. If your employer
paid you in cash and you took all of that cash to a broker and bought a single
tech stock with it, you would at least notice that you had done something.

I want to be careful about what this does and does not show, and careful with
affection rather than apology, because the limits are interesting in their own
right. This is one stock, one historical path, fifty-six events, and the
long-horizon windows overlap so heavily that the effective sample is
considerably smaller than the count makes it look. No bootstrap can manufacture
history that did not happen; resampling the same decade more cleverly does not
hand you a second decade. The backtest cannot tell you what META will do next.
It can only tell you whether any of the stories people trade at lunch survived
contact with the one history we actually have. None of them did.

I will admit the result pleased me. I suspected the boring answer all along and
built the machine to earn the right to believe it. But there is a trap sitting
inside that sentence, and it took me a while to see it: wanting the null to win
is a bias too. If you secretly root for the boring answer, you can build a
courtroom where the boring answer always wins, and you will never catch
yourself doing it, because every verdict will keep agreeing with you. The
frozen rule protected the result from my cleverness. It also had to protect the
result from my humility, and I am not sure I would believe my own conclusion
otherwise.

The best thing the null produced was a better question than the one I started
with. Every arm was graded on all fifty-six events, including the ones where a
conditional arm chose to sit still, and sitting still scores exactly zero, so
the arms I most wanted to test were judged largely on their silence. The
natural next read is conditional on firing: how does the earnings-reaction rule
do on the events where it actually acted? I cannot promote that to a real test.
I thought of it after seeing the verdict, which makes it exploratory forever, a
hypothesis for somebody else's pre-registration or for my next one. But that is
the quiet gift of a rule with teeth. Even its blind spots are informative, and
a rule that can be wrong in a specific, visible way tells you precisely which
experiment to run next.

Four times a year the shares still land. The argument, for the first time since
2012, is over. I set out to buy an edge and came home with four decisions a
year instead, which is not what I was shopping for and is worth considerably
more.
