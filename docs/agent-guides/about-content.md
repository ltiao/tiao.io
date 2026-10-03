# About content

The About Me material is intentionally split across four content sections and
one easter-egg blurb. Preserve the five boundaries, do not consolidate or move
text between them, and do not “fix” the easter egg. Use subtle dry deadpan,
understated over zany, without undercutting achievements; recognition should
read as factual observation rather than a personal brag.

| # | Location | Purpose |
|---|---|---|
| 1 | `content/_index.md` → `resume-biography-3` block `text:` | Bio basics (homepage About) |
| 2 | `content/_index.md` → markdown block "My Research" | Research mission + collaboration invite |
| 3 | `content/experience.md` → top block "Career Summary" | LinkedIn-style overview |
| 4 | `content/experience.md` → block "Collaborators" | Mentor/colleague acknowledgements |
| 5 | `data/authors/me.yaml` → `bio:` field | Humorous easter egg (NOT a placeholder) |

## Required boundaries

1. **Homepage About:** crisp, 2–3 sentences maximum: name → role → workplace
   → location → one-line research focus. Exclude PhD, supervisors, and
   recognition; sections 2 and 3 own those.
2. **Homepage “My Research”:** open exactly with **“My research is in…”**;
   tag-link key terms via `/tags/<slug>/`; end the research paragraph with one
   dry reception sentence, such as “Some of this work has appeared as Orals
   and Spotlights at NeurIPS and ICML.” Close with a brief collaboration invite
   linked to `#contact`. Do not add a PhD/supervisors paragraph.
3. **Career Summary:** first-person LinkedIn profile-summary style, not outdated
   (avoid “over X years”). Use two paragraphs: (a) current position, work
   focus, PhD, and Oral/Spotlight recognition; (b) trajectory NICTA → Data61 →
   industrial PhD appointments at Amazon Berlin/Cambridge UK + Secondmind
   Cambridge. This owns comprehensive recognition/supervisor data.
4. **Collaborators:** open exactly with *“I owe a great deal to the people
   below — colleagues and mentors who shaped how I approach research.”* Then
   use `**Name** — Title, Organization · [personal site]`; skip lineage notes,
   organization mission statements, and expansive role descriptions. Link
   personal sites where they exist.
5. **Easter-egg bio:** `data/authors/me.yaml` `bio:` is deliberate humor, not a
   placeholder. Preserve this exact value:

> My name is Louis Tiao, and I graduated from one of Australia's top engineering schools with really good grades. Now, I'm using my knowledge to help up-and-coming tech companies make it in this competitive world.

The rationale is the dry-deadpan joke that the reader knows Louis works at
Meta: Big Tech is reframed as a plucky underdog, prompting “wait, *Meta*?”.
Deleting or rewriting it destroys the joke. It appears on `/authors/me/`, while
the homepage About `text:` override supersedes it.
