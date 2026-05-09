# Content Migration Implementation Plan (Sections A–F)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate all real content from `~/Projects/tiao.io` (legacy HugoBlox `theme-academic-cv`) into `~/Projects/lively-pioneer` (new HugoBlox `kit/templates/academic-cv` v0.12.0), restructuring the homepage and menu to match the design spec. Local site renders cleanly via `hugo server` with all collections, links, and assets working.

**Architecture:** Sectional migration with verification gate after each section. Bulk operations done via inline shell scripts (idempotent where possible) — sed-driven front-matter rewrites and folder renames. Subjective parts (bio restructure, prose passes per post/pub/project) interactive. One commit per task; rollback is `git revert`.

**Tech Stack:** Hugo 0.159.2 (extended), HugoBlox `kit/templates/academic-cv` v0.12.0, Go modules, pnpm, GitHub Pages target. Markdown content with TOML/YAML front-matter.

**Spec:** `docs/superpowers/specs/2026-05-09-content-migration-design.md`

**Out of scope (covered by a follow-up plan):** Section G (repo rename + backup), Section H (deploy + DNS cutover).

---

## Pre-flight

These checks happen once before any task. Working directory is `~/Projects/lively-pioneer`.

- [ ] **Step P1: Verify dev server runs cleanly on the unmodified target**

```bash
cd ~/Projects/lively-pioneer
hugo server --bind 127.0.0.1 --port 1313 2>&1 | head -30
```

Expected: Hugo builds without errors; server reports `Web Server is available at http://127.0.0.1:1313/`. Open in browser, confirm placeholder homepage renders. Stop server (Ctrl+C).

If errors: fix Hugo install / module fetch issues before proceeding (`hugo mod get -u`, `hugo mod tidy`). Do not start the migration with a broken baseline.

- [ ] **Step P2: Confirm clean working tree on a known-good branch**

```bash
git status --short
git log --oneline -3
```

Expected: pre-existing uncommitted modifications (`M content/_index.md`, `M pnpm-lock.yaml`) and untracked files (`.npmrc`, `go.sum`) are present from the user's earlier setup. These are unrelated to migration and stay untouched.

The migration does NOT touch `pnpm-lock.yaml`, `go.sum`, `.npmrc`. The pre-existing `content/_index.md` modification will be folded into Task 1.5 (homepage edits). Note its current diff before overwriting:

```bash
git diff content/_index.md > /tmp/preexisting-index-diff.txt
wc -l /tmp/preexisting-index-diff.txt
```

If the diff contains user-specific edits we should preserve, surface them when editing `content/_index.md` in Task 1.5.

- [ ] **Step P3: Snapshot source paths for reference throughout the plan**

```
SRC=~/Projects/tiao.io
TGT=~/Projects/lively-pioneer
```

These shorthand vars are used in subsequent shell snippets.

---

## Task 1: Author profile + homepage (Section A)

**Files:**
- Create: `content/authors/me/_index.md`
- Possibly create: `content/authors/me/avatar.{jpg,png,webp}` (if user supplies one — source has none)
- Modify: `content/_index.md` (homepage blocks, `page_type`, add Featured Projects, remove redundant blocks)
- Modify: `config/_default/menus.yaml` (add Projects, Contact entries; reorder)
- Delete: `content/authors/_index.md` (target's empty placeholder)

### 1.1 Copy and audit author profile

- [ ] **Step 1.1.1: Copy source author file into the new `me` folder**

```bash
mkdir -p $TGT/content/authors/me
cp $SRC/content/authors/admin/_index.md $TGT/content/authors/me/_index.md
```

Expected: `$TGT/content/authors/me/_index.md` exists.

- [ ] **Step 1.1.2: Update author folder reference in front-matter**

Open `content/authors/me/_index.md`. The source uses `admin` only as the folder name; the inline `_index.md` does not contain `admin` references in front-matter. No edit needed at this step.

- [ ] **Step 1.1.3: Audit which structured fields are populated**

```bash
grep -E "^(experience|skills|awards|languages):" $TGT/content/authors/me/_index.md
```

Expected: lists which structured fields exist. Likely sparse (source profile predates these fields in HugoBlox v5+).

For each missing structured field that drives the `/experience/` page (`experience`, `skills`, `awards`, `languages`), add a stub with a TODO comment and surface to the user for content. Example stub format:

```yaml
# Experience. To enable, add the field below and populate.
# experience:
#   - title: Research Scientist
#     company: Meta — Central Applied Science (CAS)
#     company_url: ''
#     location: New York City
#     date_start: '2024-08-01'
#     date_end: ''
#     description: Adaptive Experimentation. Bayesian optimization, GPs, sample-efficient ML.
```

Do NOT block on this audit — capture which fields need user input and continue. The `/experience/` page will render empty sections until user fills them in; that is acceptable for the first build.

- [ ] **Step 1.1.4: Delete the empty target author placeholder**

```bash
rm $TGT/content/authors/_index.md
```

Expected: only `$TGT/content/authors/me/_index.md` remains under `$TGT/content/authors/`.

### 1.2 Restructure bio body

- [ ] **Step 1.2.1: Read full source bio and identify the natural break**

Read `$TGT/content/authors/me/_index.md`. The body (after the closing `---` of front-matter) is the long bio. It contains:
1. An intro sentence about who Louis is + research focus areas (link-heavy).
2. PhD background (Sydney, Bonilla, Ramos).
3. NeurIPS/ICML recognition.
4. (Possibly) a paragraph about joining Meta in NYC.

Identify the dividing line between "core About" (#1, #2 condensed) and "My Research" (#3, deeper research description).

- [ ] **Step 1.2.2: Replace bio body with the short About**

Use the Edit tool to replace the bio body (everything after the front-matter closing `---`) with a 3–4 sentence core About. Draft:

```markdown
Hi. I'm Louis Tiao, a research scientist at Meta's Central Applied Science (CAS) team in New York City, working on adaptive experimentation. My research is in probabilistic machine learning — approximate Bayesian inference, Gaussian processes, and Bayesian optimization. I obtained my PhD at the University of Sydney, advised by Edwin Bonilla and Fabio Ramos.
```

This becomes the body of `content/authors/me/_index.md` and is surfaced under the `resume-biography-3` block on the homepage.

The longer "My Research" content will be moved into the homepage `markdown` block in Task 1.5 — note it down before discarding from this file.

- [ ] **Step 1.2.3: User review checkpoint**

Render `hugo server`, open `/`, confirm the short About reads well under the hero. If the user wants different framing, iterate here before continuing.

### 1.3 Avatar

- [ ] **Step 1.3.1: Check for source avatar**

```bash
ls $SRC/content/authors/admin/ | grep -iE 'avatar|profile|photo'
```

Expected: empty (source has no avatar file in the folder).

- [ ] **Step 1.3.2: Surface avatar gap to user**

Ask user: "No avatar found in source `authors/admin/`. Provide a JPG/PNG/WebP for `content/authors/me/avatar.{ext}`, or proceed with the template's default initial?"

Do not block — proceed without avatar if user defers. Add a follow-up note to the gaps list in Task 6.

### 1.4 Update menu

- [ ] **Step 1.4.1: Edit `config/_default/menus.yaml`**

Replace the `main:` block contents with:

```yaml
main:
  - name: Home
    url: /
    weight: 10
  - name: Posts
    url: /#posts
    weight: 20
  - name: Talks
    url: /#talks
    weight: 30
  - name: Publications
    url: publications/
    weight: 40
  - name: Projects
    url: projects/
    weight: 45
  - name: Experience
    url: experience/
    weight: 50
  - name: Contact
    url: /#contact
    weight: 60
```

- [ ] **Step 1.4.2: Verify menu renders**

```bash
hugo server &
sleep 3
curl -s http://127.0.0.1:1313/ | grep -oE 'href="[^"]*"' | grep -iE 'posts|talks|projects|publications|experience|contact' | head -20
kill %1
```

Expected: all 7 menu links present in homepage HTML.

### 1.5 Update homepage blocks

- [ ] **Step 1.5.1: Read current homepage and identify existing blocks**

Read `$TGT/content/_index.md`. Note current block IDs and order, especially the pre-existing user modifications (captured in Step P2).

- [ ] **Step 1.5.2: Apply changes**

Make these targeted edits via the Edit tool:

1. **Add the "My Research" markdown block content** (Block 2). Replace the placeholder text (currently mentions "Moonshot team at DeepMind") with content derived from the long source bio:

```yaml
  - block: markdown
    content:
      title: 'My Research'
      subtitle: ''
      text: |-
        I work on probabilistic machine learning, with a particular focus on
        approximate Bayesian inference and Gaussian processes, and their
        applications to Bayesian optimization and graph representation
        learning. Our research has been recognized at NeurIPS and ICML, where
        it has been selected for oral and spotlight presentations.

        At Meta, I work in the Adaptive Experimentation arm of the Central
        Applied Science (CAS) team, advancing the frontiers of Bayesian
        optimization, Gaussian processes, and sample-efficient ML for
        decision-making under uncertainty.

        Reach out via email or the contact form below — collaborations welcome.
    design:
      columns: '1'
```

(Adjust prose with user input during interactive review — the above is the initial draft.)

2. **Update `page_type: blog` to `page_type: posts` in both `collection` blocks** (`id: news` and `id: posts`):

```bash
sed -i 's|page_type: blog|page_type: posts|g' $TGT/content/_index.md
```

3. **Reorder/add blocks to match spec order:**
   - Block 5: `collection id: posts` (currently exists, was Block 5 in old; stays)
   - Block 6: `collection id: talks` (currently exists; stays)
   - Block 7: **NEW** `collection id: projects` (Featured Projects)
   - Block 8: `contact-info id: contact` (currently exists; stays)

   **Remove:**
   - The `block: resume-experience id: experience` (moved fully to `/experience/`).
   - The `block: collection` with `view: citation` and `count: 0` for Publications (full list moved to `/publications/`).

4. **Insert the new Featured Projects block** between Talks and Contact:

```yaml
  - block: collection
    id: projects
    content:
      title: Featured Projects
      count: 3
      filters:
        folders:
          - projects
        featured_only: true
    design:
      view: article-grid
      columns: 3
      fill_image: false
      show_date: false
      show_read_time: false
      show_read_more: false
```

- [ ] **Step 1.5.3: Build and verify homepage**

```bash
cd $TGT && hugo --minify 2>&1 | tail -20
```

Expected: build completes; warnings about empty collections (no posts/projects yet) are acceptable.

- [ ] **Step 1.5.4: Run dev server and spot-check**

```bash
hugo server &
sleep 3
curl -s http://127.0.0.1:1313/ -o /tmp/home.html
grep -c '<section' /tmp/home.html
grep -oE 'id="[a-z]+"' /tmp/home.html | sort -u
kill %1
```

Expected: section count consistent with 8 blocks; section IDs include `posts`, `talks`, `projects`, `contact`.

### 1.6 Commit

- [ ] **Step 1.6.1: Commit author profile + homepage + menu**

```bash
cd $TGT
git add content/authors/ content/_index.md config/_default/menus.yaml
git status --short
git commit -m "feat(content): migrate author profile and restructure homepage

- Move admin author profile to authors/me/ with restructured short bio
- Add 'My Research' markdown block on homepage with content from old bio
- Add Featured Projects collection block on homepage
- Remove redundant exhaustive Experience and Publications blocks
  (now reachable via /experience/ and /publications/ subpages)
- Update homepage page_type: blog -> posts
- Add Projects (weight 45) and Contact (weight 60) menu entries"
```

---

## Task 2: Publications (Section B)

**Files:**
- Delete: `content/publications/{conference-paper,journal-article,preprint}/`
- Create: `content/publications/<slug>/` × 10 (from source)
- Modify: `content/publications/_index.md` (only if source had a custom intro worth preserving — likely no edit)

### 2.1 Drop placeholder publications

- [ ] **Step 2.1.1: List target placeholder pubs and confirm with user**

```bash
ls $TGT/content/publications/ | grep -v '^_'
```

Expected: `conference-paper`, `journal-article`, `preprint`. If user wants to retain any, skip them in 2.1.2.

- [ ] **Step 2.1.2: Delete confirmed placeholders**

```bash
cd $TGT
git rm -r content/publications/conference-paper content/publications/journal-article content/publications/preprint
```

### 2.2 Bulk migrate source publications

- [ ] **Step 2.2.1: Copy all source publication folders**

```bash
ls $SRC/content/publication/ | grep -v '^_'
```

Expected: 10 entries (`async-multi-fidelity-hpo`, `batch-bore-guarantees`, `bore-2`, `cycle-bayes`, `phd-thesis`, `spherical-features-gaussian-process`, `vi-gcn-2`, plus any others).

```bash
for d in $SRC/content/publication/*/; do
  name=$(basename "$d")
  cp -r "$d" $TGT/content/publications/
done
ls $TGT/content/publications/ | grep -v '^_'
```

Expected: all 10 source pubs now under target `publications/`.

- [ ] **Step 2.2.2: Mechanical front-matter rewrites**

For each migrated pub:
1. Rewrite author refs.
2. Rewrite `relref` shortcodes.
3. Add legacy URL alias `/publication/<slug>/`.

Per-file shell:

```bash
cd $TGT/content/publications
for slug in $(ls -d */); do
  slug=${slug%/}
  if [[ "$slug" == "_index.md" ]]; then continue; fi
  f="$slug/index.md"
  [[ -f "$f" ]] || continue
  # 1. Author rewrite
  sed -i 's/- admin$/- me/g; s/^authors: \[admin\]/authors: [me]/g' "$f"
  # 2. relref rewrites
  sed -i 's|relref "/post/|relref "/posts/|g; s|relref "/publication/|relref "/publications/|g; s|relref "/project/|relref "/projects/|g; s|relref "/event/|relref "/events/|g' "$f"
  # 3. Legacy alias
  if grep -q '^aliases:' "$f"; then
    if ! grep -q "^  - /publication/$slug/" "$f"; then
      sed -i "/^aliases:/a\\  - /publication/$slug/" "$f"
    fi
  else
    sed -i "/^---$/{x;/^$/{x;a\\
aliases:\\
  - /publication/$slug/
;b};x}" "$f"
  fi
done
```

(The `aliases:` insertion is fragile across sed dialects. If the inline awk/sed approach proves brittle in practice, fall back to a per-file Python edit using the front-matter library. Pin this as a known watch-point during execution.)

- [ ] **Step 2.2.3: Spot-check one rewritten file**

```bash
head -20 $TGT/content/publications/bore-2/index.md
grep -E '^(authors|aliases|publication_types):' $TGT/content/publications/bore-2/index.md
```

Expected: `authors:` shows `- me`, `aliases:` includes `/publication/bore-2/`, `publication_types: ['paper-conference']` (already correct).

### 2.3 Build and verify

- [ ] **Step 2.3.1: Build**

```bash
cd $TGT && hugo --minify 2>&1 | tail -20
```

Expected: 0 errors. Warnings about missing `relref` targets may appear if a publication links to a post that is not yet migrated — note these in a `/tmp/migration-warnings.log` and resolve in Task 4 once posts are in place.

- [ ] **Step 2.3.2: Spot-check pub renders**

```bash
hugo server &
sleep 3
curl -sI http://127.0.0.1:1313/publications/bore-2/ | head -3
curl -sI http://127.0.0.1:1313/publication/bore-2/ | head -3  # alias
kill %1
```

Expected: first returns `200`. Second returns `200` (Hugo writes alias HTML with `meta refresh`).

### 2.4 Featured publications

- [ ] **Step 2.4.1: Identify featured pubs (interactive)**

Ask user: "Which 6 publications should appear on the homepage Featured Publications block?" Default suggestion: BORE-2 (ICML2021), batch-bore-guarantees, async-multi-fidelity-hpo, vi-gcn-2, cycle-bayes, spherical-features-gaussian-process. User confirms or replaces.

- [ ] **Step 2.4.2: Set `featured: true` on chosen pubs**

For each chosen slug:

```bash
sed -i 's/^featured: false$/featured: true/' $TGT/content/publications/<slug>/index.md
```

If the field is missing entirely, add it:

```bash
grep -q '^featured:' $TGT/content/publications/<slug>/index.md || \
  sed -i '/^---$/,/^---$/{ /^date:/a\
featured: true
}' $TGT/content/publications/<slug>/index.md
```

- [ ] **Step 2.4.3: Verify homepage shows them**

```bash
hugo server &
sleep 3
curl -s http://127.0.0.1:1313/ | grep -A2 'id="publications"' | head -20
kill %1
```

Expected: 6 publication titles appear in the Featured Publications block.

### 2.5 Prose pass

- [ ] **Step 2.5.1: Walk each migrated pub interactively**

For each of the 10 migrated pubs (in order: BORE-2, batch-bore-guarantees, async-multi-fidelity-hpo, ...): read the abstract and summary; propose copy-edits for typos, dead links, dated affiliations. User accepts/rejects each.

This is interactive — do NOT batch. After each pub, commit individually:

```bash
git add content/publications/<slug>/
git commit -m "edit(pub): copy-edit <slug>"
```

If a pub has zero changes after review, no commit.

### 2.6 Commit Task 2 closing

- [ ] **Step 2.6.1: Commit the bulk migration + featured flags**

(If 2.5 did per-pub commits, this commit is just the structural changes from 2.1, 2.2, 2.4.)

```bash
git add content/publications/
git commit -m "feat(content): migrate publications from tiao.io

- Drop 3 placeholder publications
- Import 10 publications from tiao.io with author rewrites,
  relref normalization, and legacy /publication/<slug>/ aliases
- Mark featured publications for homepage block"
```

---

## Task 3: Projects (Section C)

Mirrors Task 2 structure (publications → projects). Source has 4 projects.

**Files:**
- Delete: `content/projects/{pandas,pytorch,scikit}/`
- Create: `content/projects/<slug>/` × 4

### 3.1 Drop placeholder projects

- [ ] **Step 3.1.1: List and confirm**

```bash
ls $TGT/content/projects/ | grep -v '^_'
```

Expected: `pandas`, `pytorch`, `scikit`.

- [ ] **Step 3.1.2: Delete**

```bash
cd $TGT
git rm -r content/projects/pandas content/projects/pytorch content/projects/scikit
```

### 3.2 Bulk migrate source projects

- [ ] **Step 3.2.1: Copy**

```bash
for d in $SRC/content/project/*/; do
  cp -r "$d" $TGT/content/projects/
done
ls $TGT/content/projects/
```

Expected: `gaussian-process-2d-hyperparameters`, `gp-sample-fourier-decomposition-3`, `example`, `external-project`, plus any others.

- [ ] **Step 3.2.2: Mechanical rewrites**

Same script as Task 2.2.2, replacing the `relref "/project/"` line and the alias path:

```bash
cd $TGT/content/projects
for slug in $(ls -d */); do
  slug=${slug%/}
  f="$slug/index.md"
  [[ -f "$f" ]] || continue
  sed -i 's/- admin$/- me/g; s/^authors: \[admin\]/authors: [me]/g' "$f"
  sed -i 's|relref "/post/|relref "/posts/|g; s|relref "/publication/|relref "/publications/|g; s|relref "/project/|relref "/projects/|g; s|relref "/event/|relref "/events/|g' "$f"
  if grep -q '^aliases:' "$f"; then
    grep -q "^  - /project/$slug/" "$f" || sed -i "/^aliases:/a\\  - /project/$slug/" "$f"
  else
    sed -i "0,/^---$/{//!b};/^---$/a\\
aliases:\\
  - /project/$slug/
" "$f"
  fi
done
```

- [ ] **Step 3.2.3: Drop the `example` and `external-project` placeholders**

These are template demo projects in source — confirm with user, then:

```bash
cd $TGT
git rm -r content/projects/example content/projects/external-project
```

(Skip this step if user wants to keep them.)

### 3.3 Featured projects

- [ ] **Step 3.3.1: Set `featured: true` on the chosen 3**

Default: `gaussian-process-2d-hyperparameters`, `gp-sample-fourier-decomposition-3`, plus one more if user has another to surface. Confirm with user.

```bash
for slug in gaussian-process-2d-hyperparameters gp-sample-fourier-decomposition-3; do
  f=$TGT/content/projects/$slug/index.md
  if grep -q '^featured:' "$f"; then
    sed -i 's/^featured: false$/featured: true/' "$f"
  else
    sed -i '/^---$/a\featured: true' "$f"
  fi
done
```

### 3.4 Build, prose pass, commit

- [ ] **Step 3.4.1: Build**

```bash
cd $TGT && hugo --minify 2>&1 | tail -10
```

Expected: 0 errors.

- [ ] **Step 3.4.2: Verify homepage Featured Projects block populated**

```bash
hugo server &
sleep 3
curl -s http://127.0.0.1:1313/ | grep -A5 'id="projects"' | head -20
kill %1
```

Expected: featured project titles appear.

- [ ] **Step 3.4.3: Verify /projects/ subpage**

```bash
curl -sI http://127.0.0.1:1313/projects/ | head -1
```

Expected: `200`.

- [ ] **Step 3.4.4: Prose pass per project (interactive)**

Same protocol as Task 2.5: walk each project's description with user, copy-edit, commit per project.

- [ ] **Step 3.4.5: Final commit for Task 3**

```bash
git add content/projects/
git commit -m "feat(content): migrate projects from tiao.io

- Drop 3 placeholder projects
- Import N projects from tiao.io (drop demo example/external-project)
- Mark 3 featured projects for homepage block"
```

---

## Task 4: Posts (Section D)

This is the largest task. 23 source posts, with some prose editing per post. Organized as: rename target folder, bulk mechanical migration, then prose passes in batches of ~5.

**Files:**
- Rename: `content/blog/` → `content/posts/`
- Delete: 5 placeholder posts under `content/posts/`
- Create: `content/posts/<slug>/` × 23 (some are folders with `index.md`, two are flat `.md` files)

### 4.1 Rename target folder + clean placeholders

- [ ] **Step 4.1.1: Rename `blog/` to `posts/`**

```bash
cd $TGT
git mv content/blog content/posts
ls content/posts/
```

Expected: `data-visualization`, `notebook-onboarding`, `project-management`, `second-brain`, `teach-courses`, `_index.md`.

- [ ] **Step 4.1.2: Drop placeholder posts**

```bash
cd $TGT
git rm -r content/posts/data-visualization \
         content/posts/notebook-onboarding \
         content/posts/project-management \
         content/posts/second-brain \
         content/posts/teach-courses
ls content/posts/
```

Expected: only `_index.md` remains.

### 4.2 Bulk migrate source posts

- [ ] **Step 4.2.1: Copy all source post folders and standalone .md files**

```bash
cd $SRC/content/post
# Folders
for d in */; do
  name=${d%/}
  if [[ "$name" == "_index.md" ]]; then continue; fi
  cp -r "$d" $TGT/content/posts/
done
# Standalone .md files (e.g. normalizing-flows.md, variational-inference-...md)
for f in *.md; do
  if [[ "$f" == "_index.md" ]]; then continue; fi
  cp "$f" $TGT/content/posts/
done
ls $TGT/content/posts/ | grep -v '^_' | wc -l
```

Expected: 23.

- [ ] **Step 4.2.2: Apply mechanical rewrites to every post**

```bash
cd $TGT/content/posts
process_post() {
  local f=$1
  local slug=$2
  sed -i 's/- admin$/- me/g; s/^authors: \[admin\]/authors: [me]/g' "$f"
  sed -i 's|relref "/post/|relref "/posts/|g; s|relref "/publication/|relref "/publications/|g; s|relref "/project/|relref "/projects/|g; s|relref "/event/|relref "/events/|g' "$f"
  # Legacy alias
  if grep -q '^aliases:' "$f"; then
    grep -q "^  - /post/$slug/" "$f" || sed -i "/^aliases:/a\\  - /post/$slug/" "$f"
  else
    awk -v slug="$slug" '
      BEGIN{c=0}
      /^---$/{c++; print; if(c==1){print "aliases:"; print "  - /post/" slug "/"}; next}
      {print}
    ' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
  fi
}

# Folder-based posts
for d in */; do
  slug=${d%/}
  [[ -f "$slug/index.md" ]] || continue
  process_post "$slug/index.md" "$slug"
done

# Flat .md posts (slug = filename without .md)
for f in *.md; do
  [[ "$f" == "_index.md" ]] && continue
  slug=${f%.md}
  process_post "$f" "$slug"
done
```

- [ ] **Step 4.2.3: Spot-check 3 rewritten posts**

```bash
for slug in normalizing-flows polya-gamma-basic-relationships getting-started; do
  echo "=== $slug ==="
  if [[ -f $TGT/content/posts/$slug.md ]]; then
    head -15 $TGT/content/posts/$slug.md
  else
    head -15 $TGT/content/posts/$slug/index.md
  fi
done
```

Expected: each shows `authors: - me` and an `aliases:` entry containing `/post/<slug>/`.

### 4.3 First build — surface errors

- [ ] **Step 4.3.1: Build with verbose output**

```bash
cd $TGT && hugo --minify 2>&1 | tee /tmp/post-build.log
```

Expected: build completes (possibly with warnings). Common issues to expect and fix immediately:

- **Missing `relref` targets** (a post links to another post by old slug): `grep -E 'REF_NOT_FOUND|cannot find' /tmp/post-build.log`. Fix: update the relref to the new path or remove the link if target wasn't migrated.
- **Missing image references** (e.g., `featured.jpg` referenced but folder is flat .md): convert flat .md posts to folder format (`mkdir <slug>/ && mv <slug>.md <slug>/index.md`) only for ones that need bundled assets.
- **Shortcode errors** (e.g. `{{< gallery >}}` from old theme not in new template): list these in `/tmp/migration-warnings.log` for manual handling per post.

- [ ] **Step 4.3.2: Resolve build-blocking errors**

For each error class, fix iteratively. Re-run `hugo --minify` after each fix. Goal: build completes successfully (warnings allowed; errors not).

- [ ] **Step 4.3.3: Verify two posts render**

```bash
hugo server &
sleep 3
curl -sI http://127.0.0.1:1313/posts/normalizing-flows/ | head -1
curl -sI http://127.0.0.1:1313/post/normalizing-flows/ | head -1   # alias
curl -sI http://127.0.0.1:1313/posts/polya-gamma-basic-relationships/ | head -1
kill %1
```

Expected: all return `200`.

### 4.4 Commit bulk migration

- [ ] **Step 4.4.1: Commit**

```bash
cd $TGT
git add content/posts/
git commit -m "feat(content): migrate 23 posts from tiao.io

- Rename content/blog/ -> content/posts/, drop 5 placeholder posts
- Import 23 posts with mechanical rewrites:
  authors [admin] -> [me]; relref /post/ -> /posts/; legacy aliases"
```

### 4.5 Prose passes (5 batches of ~5 posts each)

This is the interactive section. Walk through posts in 5 chunks:

| Batch | Posts |
|-------|-------|
| 4.5.1 | News posts: `one-paper-accepted-to-icml2023`, `one-paper-accepted-to-neurips2022`, `phd-thesis-acknowledgements`, `getting-started`, `writing-technical-content` |
| 4.5.2 | Polya-gamma series: `polya-gamma-basic-relationships`, `polya-gamma-bayesian-logistic-regression`, `polya-gamma-sigmoid-local-variational-lower-bound`, `probabilistic-matrix-factorization`, `sparse-variational-gaussian-processes` |
| 4.5.3 | DRE/info-theory: `density-ratio-estimation-for-kl-divergence-minimization-between-implicit-distributions`, `density-ratio-estimation-unsupervised-as-supervised-learning`, `tutorial-on-variational-autoencoders-with-a-concise-keras-implementation`, `variational-inference-from-an-importance-sampling-perspective`, `normalizing-flows` |
| 4.5.4 | BO/optim: `an-illustrated-guide-to-the-knowledge-gradient-acquisition-function`, `efficient-cholesky-decomposition-of-low-rank-updates`, `building-probability-distributions-with-tensorflow-probability-bijector-api` |
| 4.5.5 | Tools/practical: `docker-image-for-machine-learning-research-and-development-pytorch-jax-tensorflow`, `exploring-the-binance-cryptocurrency-exchange-api-orderbook`, `exploring-the-binance-cryptocurrency-exchange-api-recent-historical-trades`, `numpy-mgrid-vs-meshgrid`, `jupyter` |

For each batch:

- [ ] **Step 4.5.B (B = 1..5): Read each post; surface to user; apply edits**

Per post:
1. Read full content.
2. Identify candidate edits: typos; broken links; dated framing (e.g. "Starting in August I will be joining Meta" — this is now past tense); references to old site URLs.
3. For each substantive edit, surface to user with specific suggestion. Low-risk typos can be applied directly.
4. Apply approved edits.

After each post, optionally commit; or commit per batch:

```bash
git add content/posts/<slug>...
git commit -m "edit(post): copy-edit batch 4.5.B"
```

After batch complete, build to confirm no regressions:

```bash
cd $TGT && hugo --minify 2>&1 | tail -5
```

---

## Task 5: Events + slides (Section E)

**Files:**
- Delete: `content/events/example/`, `content/slides/example/`, `content/courses/`, `content/experience.md` (no — keep!)
- Create: `content/events/icml2018-tagdm/`, `content/slides/<source-slides>/`

### 5.1 Drop placeholders

- [ ] **Step 5.1.1: Drop event/slide/courses placeholders (NOT `experience.md`)**

```bash
cd $TGT
git rm -r content/events/example content/slides/example content/courses
ls content/events/ content/slides/
```

Expected: events has only `_index.md`; slides empty.

- [ ] **Step 5.1.2: Confirm `experience.md` is preserved**

```bash
ls $TGT/content/experience.md
```

Expected: file exists. Do not delete.

### 5.2 Migrate events

- [ ] **Step 5.2.1: Copy source events**

```bash
for d in $SRC/content/event/*/; do
  cp -r "$d" $TGT/content/events/
done
ls $TGT/content/events/
```

Expected: `icml2018-tagdm` (and `_index.md` from source — overwrite handled by conflict policy: keep target's `_index.md`, so move the source one out of the way first):

```bash
[[ -f $TGT/content/events/_index.md.from-source ]] || true
# If cp brought source _index.md, target's was already overwritten — restore:
git checkout HEAD -- content/events/_index.md
```

- [ ] **Step 5.2.2: Apply mechanical rewrites**

```bash
cd $TGT/content/events
for slug in icml2018-tagdm; do
  f="$slug/index.md"
  [[ -f "$f" ]] || continue
  sed -i 's/- admin$/- me/g; s/^authors: \[admin\]/authors: [me]/g' "$f"
  sed -i 's|relref "/post/|relref "/posts/|g; s|relref "/publication/|relref "/publications/|g; s|relref "/project/|relref "/projects/|g; s|relref "/event/|relref "/events/|g' "$f"
  # Legacy alias
  if grep -q '^aliases:' "$f"; then
    grep -q "^  - /event/$slug/" "$f" || sed -i "/^aliases:/a\\  - /event/$slug/" "$f"
  else
    awk -v slug="$slug" '
      BEGIN{c=0}
      /^---$/{c++; print; if(c==1){print "aliases:"; print "  - /event/" slug "/"}; next}
      {print}
    ' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
  fi
done
```

### 5.3 Migrate slides

- [ ] **Step 5.3.1: Copy source slides**

```bash
for d in $SRC/content/slides/*/; do
  cp -r "$d" $TGT/content/slides/
done
ls $TGT/content/slides/
```

Expected: source slides folder(s).

### 5.4 Verify and commit

- [ ] **Step 5.4.1: Build and check**

```bash
cd $TGT && hugo --minify 2>&1 | tail -10
hugo server &
sleep 3
curl -sI http://127.0.0.1:1313/events/icml2018-tagdm/ | head -1
kill %1
```

Expected: build clean; event page returns 200.

- [ ] **Step 5.4.2: Commit**

```bash
git add content/events/ content/slides/
git rm -r content/courses
git status --short
git commit -m "feat(content): migrate events and slides; drop courses placeholder

- Drop placeholder events/example, slides/example, courses/
- Import event icml2018-tagdm with author rewrites and legacy alias
- Import slides from tiao.io"
```

---

## Task 6: Global assets + privacy/terms + taxonomy + cleanup (Section F)

### 6.1 Merge `assets/media/`

- [ ] **Step 6.1.1: Identify which media files are referenced from migrated content**

```bash
grep -rEho 'media/[a-zA-Z0-9_./-]+' $TGT/content/ | sort -u > /tmp/refd-media.txt
wc -l /tmp/refd-media.txt
```

Expected: a list of media filenames referenced from posts/pubs/projects.

- [ ] **Step 6.1.2: Copy referenced + uncategorized media from source**

```bash
mkdir -p $TGT/assets/media
for f in $SRC/assets/media/*; do
  base=$(basename "$f")
  [[ -e $TGT/assets/media/$base ]] && continue   # skip target-existing per conflict policy
  cp -r "$f" $TGT/assets/media/
done
ls $TGT/assets/media/ | head -10
```

(Hero video `hero.mp4` is NOT used by the new template — fine to copy regardless; it's just unused asset.)

### 6.2 Copy `static/uploads/`

- [ ] **Step 6.2.1: Copy CV and other uploads**

```bash
mkdir -p $TGT/static/uploads
for f in $SRC/static/uploads/*; do
  base=$(basename "$f")
  [[ -e $TGT/static/uploads/$base ]] && continue
  cp "$f" $TGT/static/uploads/
done
ls $TGT/static/uploads/
```

Expected: `cv-louis-tiao.pdf` and any other files.

- [ ] **Step 6.2.2: Verify CV link on homepage works**

The `resume-biography-3` block has `button.url: uploads/resume.pdf` but the source CV is named `cv-louis-tiao.pdf`. Decide:
1. Rename source file: `mv $TGT/static/uploads/cv-louis-tiao.pdf $TGT/static/uploads/resume.pdf`, OR
2. Update homepage block URL: edit `content/_index.md` `button.url` to `uploads/cv-louis-tiao.pdf`.

Default: option 2 (preserve original filename for any external links). Confirm with user.

### 6.3 Privacy + Terms

- [ ] **Step 6.3.1: Copy source files**

```bash
cp $SRC/content/privacy.md $TGT/content/privacy.md
cp $SRC/content/terms.md $TGT/content/terms.md
```

If target ever ends up with its own variants (it currently doesn't), surface diff per conflict policy.

- [ ] **Step 6.3.2: Verify they render**

```bash
hugo server &
sleep 3
curl -sI http://127.0.0.1:1313/privacy/ http://127.0.0.1:1313/terms/ | head -4
kill %1
```

Expected: both 200.

### 6.4 Taxonomy audit

- [ ] **Step 6.4.1: List all tags and categories in use**

```bash
grep -rh '^- ' $TGT/content/posts/*/index.md $TGT/content/posts/*.md 2>/dev/null | \
  awk '/^tags:/{f=1;next} /^[a-z]/{f=0} f' | sort | uniq -c | sort -rn | head -30
```

(Adjust the pattern as needed — the goal is to get a tag/category histogram.)

- [ ] **Step 6.4.2: Verify each tag/category page renders**

```bash
hugo server &
sleep 3
for tag in machine-learning gaussian-processes bayesian-optimization technical news; do
  echo "$tag: $(curl -sI http://127.0.0.1:1313/tag/$tag/ | head -1)"
done
kill %1
```

Expected: all 200 (Hugo auto-generates tag pages).

- [ ] **Step 6.4.3: Surface any anomalies (typo'd tags, near-duplicates)**

If `gaussian-process` and `gaussian-processes` both exist as tags, surface to user for consolidation. Apply consolidation via sed across all content files.

### 6.5 Final build + smoke test

- [ ] **Step 6.5.1: Clean build**

```bash
cd $TGT
rm -rf public resources/_gen
hugo --minify 2>&1 | tee /tmp/final-build.log | tail -30
grep -iE 'error|warn' /tmp/final-build.log | head -20
```

Expected: 0 errors. Warnings reviewed and either resolved or accepted with note.

- [ ] **Step 6.5.2: Walk every menu link**

```bash
hugo server &
sleep 3
for url in / /#posts /#talks /publications/ /projects/ /experience/ /#contact /privacy/ /terms/; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:1313$url")
  echo "$code  $url"
done
kill %1
```

Expected: all 200.

- [ ] **Step 6.5.3: Walk a sample of legacy aliases**

```bash
hugo server &
sleep 3
for url in /post/normalizing-flows/ /publication/bore-2/ /project/gaussian-process-2d-hyperparameters/ /event/icml2018-tagdm/; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:1313$url")
  echo "$code  $url"
done
kill %1
```

Expected: all 200 (alias HTML page).

### 6.6 Final commit

- [ ] **Step 6.6.1: Commit assets, privacy/terms, any taxonomy fixes**

```bash
cd $TGT
git add assets/ static/ content/privacy.md content/terms.md content/posts/ content/publications/ content/projects/ content/events/
git status --short
git commit -m "feat(content): merge global assets and finalize taxonomy

- Copy assets/media/* from tiao.io (skip target-existing)
- Copy static/uploads/ (CV PDF)
- Add privacy.md and terms.md
- Consolidate near-duplicate tags/categories"
```

---

## Self-review

Spec coverage check (run after writing this plan):

| Spec section | Plan task | Coverage |
|---|---|---|
| Folder mapping | Tasks 2, 3, 4, 5 | ✓ All collections |
| Asset mapping | Task 6.1, 6.2 | ✓ |
| Files not migrated | Pre-flight Step P2 (untouched), implicit in scope | ✓ |
| Conflict policy | Inline in 5.2.1 (events `_index.md`), 6.1.2, 6.2.1 | ✓ |
| Front-matter rewrites (authors, relref, aliases) | Per-task scripts (2.2.2, 3.2.2, 4.2.2, 5.2.2) | ✓ |
| Author profile restructure | Task 1.1, 1.2 | ✓ |
| Homepage block edits | Task 1.5 | ✓ |
| Menu | Task 1.4 | ✓ |
| Sectional verification gates | Build + curl after each task | ✓ |
| Risks (broken relref, taxonomy sprawl, lossy rewrites) | 4.3.1, 6.4 | ✓ |
| Out of scope (theme/CSS, substantive rewrites, deploy/DNS) | Plan header explicitly defers G+H | ✓ |

**Open watch-points during execution:**
1. `aliases:` insertion via sed/awk is dialect-sensitive. If brittle, fall back to per-file Python edit using PyYAML or the toml-front-matter library.
2. Flat `.md` posts (`normalizing-flows.md`, `variational-inference-...md`) lack a folder for bundled assets; if their bodies reference local images, convert to folder format on the fly during Task 4.3.
3. Avatar gap (Task 1.3) — likely a real gap; user must supply.
4. Pre-existing `content/_index.md` modification (captured at P2) gets folded into Task 1.5 — verify nothing in the user's edits is lost.
