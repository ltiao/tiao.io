# Math rendering and validation

Global math is enabled in `config/_default/params.yaml`. This repository uses
`$...$` for inline math and `$$...$$` for display math.

## Authoring rules

- **Currency:** in prose and Markdown tables, isolate the dollar glyph as
  `<span>&#36;</span>` and keep the amount outside the span—for example,
  `<span>&#36;</span>18`. A bare `$` starts inline math, and in this stack `\$`
  is still consumed before KaTeX runs. A bare `&#36;` can be decoded back to a
  delimiter before auto-rendering; the span creates the required boundary.
  Currency must remain prose/accessibility text, not KaTeX.
- **Inline math:** pair every `$` delimiter within the same paragraph or table
  cell. Do not use math delimiters merely to typeset currency. Write comparison
  operators as TeX commands (`\lt`, `\gt`, `\leq`, `\geq`) rather than literal
  `<` or `>` characters; in particular, a literal `<` can start an HTML tag and
  truncate the inline expression before KaTeX sees it.
- **Display math:** place paired `$$` delimiters around one expression. Never
  begin an interior TeX line with a Markdown block marker such as `+`, `-`,
  `*`, `>`, or `#`; Markdown can claim the line before KaTeX sees it. Keep a
  short expression on one line or use an alignment environment:

  ```latex
  $$
  \begin{aligned}
  D(R) &= a(R) \\
       &\quad + b(R) \\
       &\quad + c(R).
  \end{aligned}
  $$
  ```

- **Tables:** inspect every cell containing math or currency. Markdown table
  separators and TeX punctuation can interact before rendering, so visual DOM
  output—not the source alone—is authoritative.

Before calling a rendering issue fixed, check delimiter pairing and use this
two-tier workflow.

## Tier 1: targeted edit loop

Use the existing Hugo development server, or run
`hugo server --disableFastRender --buildDrafts`. In a named `agent-browser`
session, reload the affected page and inspect the edited passage at the current
working viewport. Confirm the target prose, table, or display has rendered and
that no raw delimiter or literal TeX is visible. This is rapid feedback, not
the completion gate.

## Tier 2: completion gate

1. Run `pnpm run build` and serve the generated `public/` directory with a
   temporary static server. Validate the production output, not only Hugo's
   development server.
2. Open the target URL in a descriptive named session, wait for network idle,
   and check 1440×900, 390px, and 320px viewports. Representative commands are
   `agent-browser --session math-rendering open <URL>`,
   `agent-browser --session math-rendering wait --load networkidle`, and
   `agent-browser --session math-rendering set viewport <WIDTH> <HEIGHT>`.
3. At every viewport, capture the affected region, inspect an accessibility
   snapshot, and check `console` plus `errors`. Verify currency is exposed as
   prose, and each display is one math expression—not raw `$$`, a Markdown
   list, or literal TeX.
4. At 320px, require both page-level and per-display math containment. A page
   can have no global overflow while KaTeX silently makes one display
   horizontally scrollable, which then also requires keyboard focus. Evaluate
   both widths and every display:

   ```js
   ({
     viewport: window.innerWidth,
     document: document.documentElement.scrollWidth,
     displays: [...document.querySelectorAll('.katex-display')].map((node) => ({
       client: node.clientWidth,
       scroll: node.scrollWidth,
     })),
   })
   ```

   Require `viewport === document` and `client === scroll` for every display.
   If a display is too wide, split it across aligned rows and rerun the
   accessibility audit; do not rely on a pointer-only horizontal scroller.

5. Record fresh evidence when resolving a defect. Close the named browser
   session and temporary static server when validation ends.

Hugo's plain development server does not run the production Pagefind step, so a
`/pagefind/pagefind.js` failure there is an expected dev limitation, not a
product bug. Keep this distinction accurate in `dogfood-output/report.md`.

Routine checks may remain ephemeral. Retain existing dogfood artifacts; persist
new screenshots/reports only when documenting a defect, when the user requests
them, or when a regression needs durable evidence. Update issue counts and
statuses accurately. Close temporary browser sessions and static-preview
servers when validation ends; leave a development server running only when the
user requested it.
