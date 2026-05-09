# Repo conventions for coding agents

## Author stubs for non-`me` first authors

When importing a new publication, talk, or post whose first author is **not** `me`,
add a stub for that author at `data/authors/<slug>.yaml`. Without a stub, the
HugoBlox v0.12.0 `card` view (used on the homepage Featured Publications grid
and elsewhere) urlizes the author name, fails to find a profile, and falls back
to rendering the slug verbatim — e.g. `jihao-andreas-lin` instead of
`Jihao Andreas Lin`.

The slug is the urlized form of the display name (`"Jihao Andreas Lin"` →
`jihao-andreas-lin`). Minimum stub:

```yaml
schema: hugoblox/author/v1
slug: jihao-andreas-lin
name:
  display: Jihao Andreas Lin
```

Existing stubs live in `data/authors/`. Mirror their format. Owner profile
(`me.yaml`) is the exception — full bio, links, education, etc.

## Publication hero images

Hero images on publication pages should **never** carry a caption. Leave
`image.caption: ''` empty. Captions clutter the layout and the figure
context is already conveyed by the abstract / summary on the same page.
The figure caption from the paper belongs in the paper, not on the
publication card.

## Publication and technical-post summaries

`summary:` fields on **publications** and **technical blog posts** (category
`technical`) are written in the first-person plural — start with **"We …"**
(e.g. "We introduce…", "We extend…", "We study…", "We propose…", "We
give…"). This holds even for single-authored work like the PhD thesis.
Don't use "Our framework…", "This paper…", or impersonal descriptions
("A guide to…", "An open-source platform…"). The summary is shown on
cards, so keep it tight: 1–3 sentences.

The longer `abstract:` field on publications can be used verbatim from the
paper without rewriting.

**Exceptions:** news posts (category `news`) and personal essays
(category `personal`) follow their own register — typically "Our paper X
was accepted…" or first-person singular for personal milestones. Don't
force "We …" onto these.

## Tags

**URL pattern is `/tags/<slug>/` plural.** `taxonomies` in `config/_default/hugo.yaml`
maps `tag: tags`, so the rendered URL is plural. `/tag/<slug>/` 404s. The slug is
kebab-case of the tag name (`"Bayesian Optimization"` → `bayesian-optimization`,
`"AutoML"` → `automl`).

**Mirror tags from the canonical artifact.** When a news post announces a paper,
or a talk presents one, copy the publication's tag set onto the news/talk so a
visitor browsing a tag page sees all forms of the work together. Workshop and
conference versions of the same paper should share tags too.

**The first tag in a YAML list is shown prominently** on featured cards, so make
it the most distinctive tag — not `Machine Learning`. `Machine Learning` is a
useful filter but a poor lead descriptor.

**When introducing a new tag, check it has at least 2–3 file uses.** Singletons
clutter tag-cloud renderings. Existing tag clusters worth mirroring: see what
already exists with `grep -r '^- ' content/*/index.md | grep tags` or via a
quick histogram of `tags:` and `categories:` blocks.

## Cross-linking

Liberal internal cross-linking is encouraged. Patterns currently in use:

- **News → Publication:** italicize the paper title and link to the publication
  entry via `[*Title*]({{< relref "/publications/<slug>" >}})`.
- **Talk → Publication:** add a `Publication` entry to the talk's `links:`
  array with `icon: hero/book-open` and `url: /publications/<slug>/`. Add an
  `arrow-top-right-on-square` icon for external proceedings/preprint.
- **Project body → Publication:** for software projects with a paper (Ax, BORE,
  AutoGluon), link the paper title in the body via `{{< relref >}}`.
- **Prose → Tag pages:** in homepage and experience-page prose, link keyword
  mentions like "Bayesian optimization" and "Gaussian processes" to
  `/tags/<slug>/`. Don't link `Machine Learning` (too generic to be useful).
- **Series cross-links:** multi-part posts (e.g. Pólya-Gamma Parts I/II/III)
  should reference each other in a `> [!NOTE]` block at the top.

**`{{< relref >}}` cannot resolve draft pages** — Hugo errors `REF_NOT_FOUND`
during build. If linking from a published page to a draft, drop the relref and
either omit the link or use a plain text mention until the target is promoted.
Linking *from* a draft *to* a published page works fine.

**Icon format is prefixed** in the v0.12.0 kit. The legacy `icon: foo` +
`icon_pack: fas/fab/ai` style is silently ignored — unresolved icons render as
literal text. Use:

| Use case | Icon |
|---|---|
| Website / project home | `hero/globe-alt` |
| GitHub | `brands/github` |
| On-site publication entry | `hero/book-open` |
| External link (proceedings, blog) | `hero/arrow-top-right-on-square` |
| Download (PDF, file) | `hero/arrow-down-tray` |
| PDF | `hero/document-text` |

Heroicons (any name from heroicons.com) work under `hero/`. Brand icons under
`brands/`. Academic icons (Scholar, ORCID) under `academicons/`.

**Inline icons in body markdown** also work via the `{{< icon >}}` shortcode:
`{{< icon name="arrow-down-tray" >}}`. Default pack is `hero`, so most
shortcode usages can omit the prefix. The legacy FontAwesome-style names
(`name="download"`, `name="github"`) are the common migration trap — they
produce empty `<span>` output. Always use Hero v2 names (e.g. `download` →
`arrow-down-tray`).

There's also `{{< button >}}` for call-to-action buttons with icons:
`{{< button url="..." style="primary" icon="arrow-down-tray" >}}Download{{< /button >}}`.
Styles: `primary`, `outline`, `ghost`.

## Asking for external URLs

**Never fabricate URLs.** If a project or publication needs an external link
(GitHub release, engineering blog post, lab page, docs site, ResearchGate
profile), and you don't already have the exact URL from the user's message,
the existing repo state, or a tool result — **ask the user for it.** Don't
guess at `*.readthedocs.io` URLs, paper PDF locations, GitHub orgs, or social
profiles. A 404 link is worse than a missing one. The Anthropic system
guidance reinforces this: don't generate URLs unless you're confident they're
correct.

When the user provides a URL in chat, treat it as authoritative and add it
exactly as given.
