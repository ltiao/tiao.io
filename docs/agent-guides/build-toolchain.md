# Build toolchain

Keep the `postinstall` symlink in `package.json`:

```json
"postinstall": "ln -sf ../@tailwindcss/cli/dist/index.mjs node_modules/.bin/tailwindcss"
```

This repo uses pnpm, while Hugo's Tailwind integration expects an npm-style bin.
HugoBlox's `blox` module calls `css.TailwindCSS` from
`layouts/_partials/css.html`; it reads `node_modules/.bin/tailwindcss` directly,
requires a Node.js script, and ignores `$PATH`. A standalone/Homebrew
`tailwindcss` cannot substitute (Hugo dropped standalone-binary support).

pnpm creates a `#!/bin/sh` shim for its isolated store and Hugo aborts with:

```
binary "tailwindcss" is not a Node.js script
```

npm would link directly to `@tailwindcss/cli/dist/index.mjs`, a real
`#!/usr/bin/env node` script. The postinstall symlink reproduces that behavior
after every `pnpm install`, `pnpm add`, fresh clone, and CI build. If Tailwind
moves its CLI entry point, update the target to match the `bin` field of
`@tailwindcss/cli`.

`pnpm-workspace.yaml` deliberately sets
`allowBuilds: { '@parcel/watcher': false }`. That transitive dependency ships a
build script pnpm blocks by default. Hugo watches files itself and invokes
`tailwindcss` one-shot, so it never exercises Parcel's build; declining it only
silences pnpm's approval prompt.
