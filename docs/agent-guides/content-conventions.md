# Content conventions

## Post dates

Always include `date: YYYY-MM-DD` in a new post's frontmatter, including
posts marked `draft: true`. Use the current date unless the user specifies
another date; preserve existing dates when editing posts. Undated drafts can
render at their direct URL while remaining absent from the recent posts
listing because of chronological sorting.

## Author stubs

When importing a new publication, talk, or post whose first author is **not**
`me`, add a stub at `data/authors/<slug>.yaml`. Without one, HugoBlox v0.12.0's
`card` view urlizes the name, cannot find a profile, and renders the slug
verbatim. The slug is the urlized display name (`"Jihao Andreas Lin"` →
`jihao-andreas-lin`). Minimum stub:

```yaml
schema: hugoblox/author/v1
slug: jihao-andreas-lin
name:
  display: Jihao Andreas Lin
```

Mirror `data/authors/` stubs. `me.yaml` is the owner-profile exception: keep
its full bio, links, education, and other profile fields rather than reducing
it to the minimum stub.

## Publication presentation and summaries

Publication hero images must leave `image.caption: ''` empty. The paper's
figure caption belongs in the paper; the page abstract/summary already gives
the figure context.

`summary:` on publications and technical blog posts (category `technical`)
starts with **"We …"** (`We introduce…`, `We extend…`, `We study…`, `We
propose…`, `We give…`), including single-author work such as the PhD thesis.
Use 1–3 tight sentences; do not use `Our framework…`, `This paper…`, or
impersonal descriptions such as `A guide to…` and `An open-source platform…`.
News (`news`) and personal essays (`personal`) keep their own register:
typically `Our paper X was accepted…` or first-person singular for milestones.
The longer publication `abstract:` may be copied verbatim from the paper.

## Tags and cross-links

The URL pattern is `/tags/<slug>/` (plural), because `config/_default/hugo.yaml`
maps `tag: tags`; `/tag/<slug>/` is a 404. Slugs are kebab-case
(`"Bayesian Optimization"` → `bayesian-optimization`, `"AutoML"` → `automl`).
Mirror the canonical publication's tag set on announcing news/talks and share
tags across workshop/conference versions. The first YAML tag is prominent on
featured cards, so make it distinctive; `Machine Learning` is a useful filter,
not a lead descriptor. A new tag should have at least 2–3 file uses. Inspect
existing clusters with `rg` and a histogram of `tags:`/`categories:` blocks.

Cross-link liberally using these established forms:

- News → publication: `[*Title*]({{< relref "/publications/<slug>" >}})`.
- Talk → publication: a `Publication` link with `icon: hero/book-open` and
  `url: /publications/<slug>/`; external proceedings/preprints use
  `hero/arrow-top-right-on-square`.
- Software project → paper: link its title in the body with `{{< relref >}}`.
- Homepage/experience prose → useful keyword tag pages such as
  `/tags/<slug>/`; do not link generic `Machine Learning`.
- Multi-part posts (for example Pólya-Gamma Parts I/II/III) reference one
  another in a `> [!NOTE]` block at the top.

`{{< relref >}}` cannot resolve draft pages and causes `REF_NOT_FOUND` during
build. From a published page to a draft, omit the link or use plain text until
promotion; draft → published links work.

## Icons, buttons, and URLs

Use the v0.12.0 prefixed icon format. Legacy `icon: foo` plus
`icon_pack: fas/fab/ai` is silently ignored and renders literal text.

| Use case | Icon |
|---|---|
| Website / project home | `hero/globe-alt` |
| GitHub | `brands/github` |
| On-site publication entry | `hero/book-open` |
| External link (proceedings, blog) | `hero/arrow-top-right-on-square` |
| Download (PDF, file) | `hero/arrow-down-tray` |
| PDF | `hero/document-text` |

Heroicons (any name from heroicons.com) use `hero/`; brands use `brands/`;
academic icons (Scholar, ORCID) use `academicons/`. Inline body icons use
`{{< icon name="arrow-down-tray" >}}`; the default pack is `hero`. Legacy
FontAwesome-style names such as `download` and `github` produce empty spans;
use Hero v2 names (`download` → `arrow-down-tray`) or the appropriate prefixed
brand icon. Buttons use:
`{{< button url="..." style="primary" icon="arrow-down-tray" >}}Download{{< /button >}}`
with styles `primary`, `outline`, or `ghost`.

Never fabricate URLs. For an external link not supplied by the user, present in
the repository, or returned by a tool, ask the user for the exact URL. A user
provided URL is authoritative and must be added exactly as given.
