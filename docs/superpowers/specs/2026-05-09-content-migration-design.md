# Content migration: tiao.io → lively-pioneer

**Status:** design
**Date:** 2026-05-09
**Owner:** Louis Tiao

## Goal

Migrate the personal academic site from `~/Projects/tiao.io` (legacy HugoBlox `theme-academic-cv`, hugo-blox-builder modules) to `~/Projects/lively-pioneer` (new HugoBlox `kit/templates/academic-cv` v0.12.0, Hugo 0.159.2). Replace the live site at tiao.io with the new build, preserving canonical post URLs and content history.

Migration is full-replacement, not a curated subset: all real content carries over, with thoughtful prose editing during the migration rather than a separate cleanup pass.

## Architecture

The new template is **block-based** (`content/_index.md` lists `block:` entries) rather than widget-based. Markdown content under `post/`, `publication/`, `project/`, `event/` carries forward by renaming the top-level folder to the new pluralized convention.

### Folder mapping

| Source (`tiao.io/content/`) | Target (`lively-pioneer/content/`) | Notes |
| --- | --- | --- |
| `authors/admin/` | `authors/me/` | Adopt new template's `me` username convention. |
| `post/` | `posts/` | Rename target `blog/` → `posts/`. Update homepage `page_type: blog` → `posts`. |
| `publication/` | `publications/` | Target already pluralized. |
| `project/` | `projects/` | Target already pluralized. |
| `event/` | `events/` | Target already pluralized. |
| `slides/` | `slides/` | Same name. |
| `privacy.md`, `terms.md` | `privacy.md`, `terms.md` | Add to target. |

### Asset mapping

| Source | Target |
| --- | --- |
| `assets/media/*` | merge into target `assets/media/*` |
| `static/uploads/*` (CV PDF) | `static/uploads/*` |
| `content/authors/admin/avatar.*` | `content/authors/me/avatar.*` |
| `images/screenshot.png`, `tn.png`, `preview.png` | dropped (template marketing artifacts) |

### Files not migrated

`theme.toml`, old `go.mod`/`go.sum`, `netlify.toml`, `LICENSE.md`, `README.md`, `.editorconfig`, `.github/`, `academic.Rproj` — target keeps its own versions.

### Conflict policy

Where a file exists in both source and target, prefer the **target** (newer) version. The realistic conflicts are:

- `content/<collection>/_index.md` — target uses block-based listing; source uses old widget config. Keep target. Show diff before discarding source's version.
- Template-placeholder content in target — drop after listing for confirmation:
  - `posts/{data-visualization, notebook-onboarding, project-management, second-brain, teach-courses}`
  - `projects/{pandas, pytorch, scikit}`
  - `publications/{conference-paper, journal-article, preprint}`
  - `events/example`, `slides/example`
  - `courses/` (entire collection — no menu/homepage wiring; source has no teaching content)

## Front-matter rewrites

Most fields carry forward unchanged. Targeted rewrites:

**All content types:**
- `authors: [admin]` → `authors: [me]`.
- `projects: ["foo"]` cross-references — verify each named project still exists post-migration; drop entries pointing to dropped projects.
- Body shortcodes: `{{< relref "/post/..." >}}` → `{{< relref "/posts/..." >}}`; same for `publication`→`publications`, `project`→`projects`, `event`→`events`.
- `aliases:` entries — keep verbatim (legacy URL redirects).
- **Add new alias entries** to preserve old singular canonical URLs: every post gets `/post/<slug>/` added to its `aliases:` list; same pattern for `publication`→`/publication/<slug>/`, `project`→`/project/<slug>/`, `event`→`/event/<slug>/`. Hugo emits redirect HTML at the alias path, so direct hits on old URLs (`tiao.io/post/normalizing-flows/`) still resolve. This is the mechanism that lets us pluralize folders without breaking inbound SEO links.

**Publications:** source already uses string `publication_types` (`'paper-conference'`, etc.) matching the new template's vocabulary. Spot-check one entry; no bulk rewrite needed.

**Posts:** `featured`, `draft`, `categories`, `tags` carry verbatim. Global tag/category audit at end.

**Author profile (`authors/me/_index.md`):**
- Front-matter (status, role, organizations, interests, education, social, email, experience, skills, awards, languages) carries verbatim from source. Audit which structured fields (experience/skills/awards/languages) are populated; if sparse, fill in interactively before continuing.
- Bio body restructured into:
  - **Short core About** (~3-4 sentences) — stays as the body of `authors/me/_index.md`, surfaced by the `resume-biography-3` block.
  - **"My Research" paragraph** — a longer research-focused paragraph that goes into the `markdown` block in `content/_index.md`, replacing the placeholder text.

**Homepage `content/_index.md`:**
- `username: me` (already correct).
- `page_type: blog` (×2) → `page_type: posts`.
- Add a Featured Projects `collection` block.
- Remove the exhaustive `resume-experience id:experience` block and the count=0 `view: citation` Publications block — both are redundant with the dedicated `/experience/` and `/publications/` subpages.

## Final homepage block order

| # | Block | Purpose |
|---|-------|---------|
| 1 | `resume-biography-3` | Hero / short About |
| 2 | `markdown` | "My Research" paragraph |
| 3 | `collection` id:news | News (category=news posts) |
| 4 | `collection` id:publications | Featured Publications (count=6, `featured_only: true`) |
| 5 | `collection` id:posts | Recent Posts (count=8, category=technical) |
| 6 | `collection` id:talks | Recent & Upcoming Talks (folders=events) |
| 7 | `collection` id:projects | Featured Projects (count=3, `featured_only: true`, `view: article-grid`, columns=3) |
| 8 | `contact-info` id:contact | Contact form |

## Final menu (`config/_default/menus.yaml`)

| Name | URL | Weight |
|---|---|---|
| Home | `/` | 10 |
| Posts | `/#posts` | 20 |
| Talks | `/#talks` | 30 |
| Publications | `publications/` | 40 |
| Projects | `projects/` | 45 |
| Experience | `experience/` | 50 |
| Contact | `/#contact` | 60 |

Anchor IDs (`#posts`, `#talks`, `#contact`) match `id:` fields in homepage blocks.

## Migration sequence (sectional pass)

Each section ends with `hugo server` running cleanly and a manual browser spot-check of the affected pages before proceeding.

### Section A — Author profile + homepage

- Copy `tiao.io/content/authors/admin/_index.md` → `authors/me/_index.md` and avatar.
- Audit structured fields (experience/skills/awards/languages); fill gaps interactively.
- Restructure bio body into core About + "My Research" markdown block.
- Update homepage `page_type` and add the Featured Projects block.
- Update menu (add Projects, Contact entries; reorder).
- Verify `/`, `/experience/`, hero/avatar render.

### Section B — Publications

- List and drop placeholder pubs.
- Copy each source pub; apply front-matter rewrites.
- Prose pass: light copy-edit of abstracts/summaries (typos, dead links, outdated affiliations). Substantive changes flagged for user decision.
- Spot-check BORE-2 renders; verify `featured` flag controls homepage placement.

### Section C — Projects

- List and drop placeholder projects.
- Copy source projects; rewrite author refs.
- Prose pass on project descriptions.
- Mark 2-3 projects with `featured: true` for the homepage block.
- Verify project pages list associated posts/pubs (Hugo backlinks via `projects:` front-matter).

### Section D — Posts (the long pole)

- Rename target `blog/` → `posts/`. Drop placeholder posts.
- Copy ~30 source posts. Mechanical rewrites (authors, relrefs).
- **Prose editing in scope.** Per post: read-through, propose edits (typos, broken links, dated framing such as "Starting in August I will be joining Meta"). Batch low-risk edits (typos), queue substantive ones for explicit user decision.
- Work through in chunks of ~5 posts at a sitting.
- Each chunk: build site, verify no shortcode/relref errors.

### Section E — Events + slides

- Drop `events/example`, `slides/example`, `courses/`.
- Copy `event/icml2018-tagdm/`, slides verbatim. Light prose pass.

### Section F — Global assets + cleanup + taxonomy

- Merge `assets/media/*` (filename collisions: prefer source if it's a real referenced asset, target otherwise; show diff for any conflict).
- Copy `static/uploads/*` (CV PDF, etc.), `privacy.md`, `terms.md`.
- Final pass: build site, walk taxonomy pages, fix broken links / missing tags.

### Section G — Repo rename + backup

- Rename old GitHub repo `tiao.io` → `tiao.io-archive` (or chosen suffix); confirm GitHub Pages source removed from it.
- Rename current GitHub repo `lively-pioneer` → `tiao.io`.
- Update local clone remote URLs (`git remote set-url ...`).
- Optional but recommended: rename local working dirs (`~/Projects/tiao.io` → `~/Projects/tiao.io-archive`, then `~/Projects/lively-pioneer` → `~/Projects/tiao.io`).
- All GitHub-side actions confirmed before execution.

### Section H — Deploy + DNS

- Confirm GitHub Pages enabled on new repo (per `hugoblox.yaml: host: 'github-pages'`); set source branch.
- Move custom domain `tiao.io` from old repo's Pages config to new (remove from old first to avoid duplicate-domain block).
- Verify CNAME / `A` records still point at GitHub Pages IPs (no DNS change at registrar expected).
- Smoke test: `tiao.io`, `www.tiao.io`, sample old singular URL `tiao.io/post/normalizing-flows/` (alias-redirected to `/posts/normalizing-flows/`), one publication and project alias likewise.
- Disable Pages on archived repo to eliminate races.

## Verification gates

Per section: `hugo server` runs cleanly + manual browser spot-check before next section.

Final pre-deploy checks:
- `hugo --minify` builds with zero errors.
- All menu links resolve.
- A representative post, pub, project, event renders correctly.
- Featured publications and featured projects appear on homepage.
- Author bio + experience/skills surface correctly on `/` and `/experience/`.
- Tag/category pages render without 404s.
- CV PDF and other static uploads accessible.

## Risks

- **Lossy front-matter rewrites:** sed-style bulk rewrites can mis-match across nested front-matter fields. Mitigation: dry-run each rewrite, commit per section so rollback is one `git revert`.
- **Broken `relref` shortcodes:** old posts cross-link via `{{< relref "/post/..." >}}`. Mitigation: build with `--strict` (or fail-on-warn equivalent) at end of Section D.
- **Tag/taxonomy sprawl:** source tags accumulated over years; new template may render some inconsistently. Mitigation: Section F audits and consolidates.
- **DNS race:** custom domain on both repos at once will fail. Mitigation: explicit ordering in Section H (remove from old before adding to new).
- **Avatar/image format compatibility:** new template may expect `.webp`/`.png` instead of `.jpg`. Mitigation: verify in Section A; convert if needed.

## Out of scope

- Theme/CSS customization beyond template defaults.
- Substantive rewriting of post content beyond copy-edits + user-flagged decisions.
- Importing analytics, comment threads, or third-party integrations.
- Migrating the `public/` build artifact (regenerated from source).
