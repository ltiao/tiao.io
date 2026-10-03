# Motivating policy trees

Two static, light-only tree diagrams, one per evaluation order. Full outcome branches, conditional probabilities, stopping reasons, best retained score, cumulative converted costs, and backward expectations. Sources are `assets/diagrams/gittins/{cheap-first,larger-first}.html`; the `policy-tree` shortcode embeds their accessible SVG directly. No separate SVG/PNG exports or shared installed-skill changes.

## Style receipt

Sampled local draft: http://localhost:1313/posts/gittins-acquisition-functions/

- Paper: #fafaf7, a warm adjustment of the site's white background.
- Ink: #24292f, sampled body color.
- Accent: #085ec4, sampled primary-600; reserved for the policy-value summary.
- Muted: #4f5d75, retained diagram token for connectors.
- Title: Lora 700; body: Source Serif 4 400/600. These match the site's declared families, using its serif treatment rather than the skill's generic sans labels.
- Google Fonts sources: standalone files request Lora 700 and Source Serif 4 400/600 using the approved fonts.googleapis.com/css2 endpoint. Inline embeds inherit the site's font loading. The preview browser had no loaded Lora/Source Serif font faces; screenshots therefore validate the explicit Georgia fallback, not an exact webfont match.
- Size: custom portrait 640 × 864; five tree nodes and four connections per panel. The summary is a separate focal annotation. Full viewport width on phones keeps node labels at 12px at 320px.

## Verification

Both HTML sources pass diagram-design self_check.py. Decimal arithmetic verifies 80.4, 81.15, 78.9, and 80.95. Production build with HUGO_BUILDDRAFTS=true passes. Production preview inspected at 1440, 390, and 320px; figures fit, text stays within the SVG canvas, accessible title/description and captions render, and no browser errors were reported. The subsequent editorial pass resolved the display-equation overflows at 320px; see editorial-review.md for final whole-post validation. The user's existing development server is left running.
