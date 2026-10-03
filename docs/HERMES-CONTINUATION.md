# Blog continuation

## Start here

This snapshot preserves the authored WIP from the Mac's `tiao.io` checkout.
Start a fresh `default` conversation against this repository rather than
resuming the old long writing session. Use its imported history as searchable
reference once migration records identify the native Hermes session.

Read `AGENTS.md`, then `docs/post-plans/gittins-acquisitions/brief.md`,
`CONTEXT.md`, `diagram-notes.md`, and `editorial-review.md` in that directory.
The current Gittins draft is `content/posts/gittins-acquisition-functions/index.md`.
Ask Louis which writing task to continue before editing other drafts.

## Preserved work

The source checkout remained on `main` at `9698966f96039d8f20b9f35dcf26ac0841337210`.
Its authored state includes the canonical instruction guides, Gittins post and
diagrams, responsive equation shortcodes/styles, GP-attention posts,
Pólya-gamma drafts, DFS and prediction-market drafts, the Matérn draft, and
existing dogfood evidence. The two Vest-timing writing documents under `docs/`
were recovered from ge66 earlier; their editorial integration is still open.
Obsidian UI state and ignored dependency/build directories are not project code.

The website repository is public. A migration branch preserves work without
merging into its deployment branch, but still exposes the draft sources.
Publishing a Git branch and publishing a finished post are separate decisions.
Keep `draft: true` on the new drafts until Louis approves publication.

## Verification on 2026-10-03

The Mac's `HUGO_BUILDDRAFTS=true pnpm run build` passed, including Pagefind.
Only existing deprecation warnings were observed. Production-output browser
checks of the Gittins post passed at 1440×900, 390×900 and 320×900: 41 visible
display equations, zero KaTeX errors, zero display overflows, and document width
equal to viewport width. The inspected screenshots show readable wide/narrow
layouts. Narrow equations define `Q_1` before `V_1`. Browser errors were empty.
Temporary preview/browser processes were stopped.

These checks establish rendering of this draft, not a new mathematical review
of all posts or readiness to publish the entire WIP collection. Follow the
repository's math/rendering guide for later edits. Use the technical team for
substantive build or implementation work; do not install project dependencies
or expose a preview server outside the existing sandbox grant as a workaround.

## History provenance

The selected Codex reference is `01a07087-cd92-7272-a8a3-147d06d2eec6`, captured
from the Mac's native session store; its last recorded activity is
2026-09-07T06:36:57.691Z. It contains an imported Claude prefix and has a
foreign recorded cwd. Preserve that history as reference, use the destination
repository explicitly, and avoid database surgery or importing its duplicate
Claude source separately. The source transcript remains unchanged.
