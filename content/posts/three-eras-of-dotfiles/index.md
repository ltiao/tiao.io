---
aliases:
  - /post/three-eras-of-dotfiles/
title: Three eras of my dotfiles
subtitle: ""
summary: We tour a chezmoi-templated dotfiles setup that bootstraps a fresh machine in one command — terminal, shell, prompt, multiplexer, multi-environment templating, and the developer tool layer. We work through the design choices and the gotchas hit along the way.
date: "2026-05-20T00:00:00Z"
draft: false
featured: true
categories:
  - technical
authors:
  - me
tags:
  - Dotfiles
  - Developer Tools
  - Chezmoi
  - Shell
image:
  focal_point: Smart
  preview_only: false
projects: []
---

Earlier today I ran `chezmoi init ltiao` on my Arch laptop, expecting a fresh
copy of my dotfiles. What I got instead was an empty `chezmoi status`, an
unfamiliar `~/.zshrc`, and — after some confused poking around — a six-year-old
version of myself staring back at me from `~/.local/share/chezmoi/`. Bullet-train
theme. neofetch. Solarized Dark. All of it. I had forgotten to pass
`--branch=main`, and chezmoi had dutifully cloned `master`, which still held the
pre-chezmoi flat-file history of the same repo.

That little goof is how I ended up spending a couple of hours rebuilding my
developer environment for the third time. The result is the cleanest setup I've
had — one repo, one command to bootstrap a new machine, the same prompt and
shortcuts everywhere I work, and per-machine differences expressed declaratively
rather than as folklore in my head. This post is a tour of where it landed and
how it got here.

## Era 1: The bare-git-repo trick

The original setup, circa late 2020, was about as minimal as it gets. From the
Notion page where I kept my install notes at the time:

```bash
git clone --bare https://github.com/ltiao/dotfiles ~/dotfiles.git
alias gitdot='git --git-dir=$HOME/dotfiles.git --work-tree=$HOME'
gitdot status
```

The trick is that you keep a bare git repo at `~/dotfiles.git` and pretend `$HOME`
is its working tree. Run `gitdot add ~/.zshrc; gitdot commit; gitdot push` and
your dotfiles are versioned. On a new machine, clone the bare repo and check
out the branch — files appear in `$HOME` directly, no symlinks, no extra tools.

It's elegant for a single machine. The whole "tool" is two lines of shell. No
templating, no install scripts, no abstractions — just git, doing what git does.

The trouble starts when you have more than one machine. My personal Arch laptop
and my work MacBook both wanted slightly different `.zshrc` files. The bare-repo
trick has no answer for that: branches drift, merges conflict over inessential
differences (a `brew shellenv` line, a different plugin list), and you find
yourself maintaining a mental map of which file lives on which branch.

The 2020 stack on top of the bare repo, for the historical record:
[oh-my-zsh][omz] with the bullet-train theme, [neofetch][neofetch] on shell open,
the [kitty][kitty] terminal with a Solarized Dark theme, `pyenv` and
`virtualenvwrapper` for Python, and `~/.Xresources` for the X11 apps I still
used at the time. Artifacts of taste at a specific moment.

[omz]: https://ohmyz.sh/
[neofetch]: https://github.com/dylanaraps/neofetch
[kitty]: https://sw.kovidgoyal.net/kitty/

## Era 2: A botched first chezmoi attempt

A couple of years later I hit the multi-machine wall and decided to migrate to
[chezmoi][chezmoi]. chezmoi is a small Go program that treats a git repo as the
source of truth for your dotfiles, then applies the contents (with optional
templating) to `$HOME` on whatever machine you're on. The exact problem the
bare-repo trick couldn't handle.

[chezmoi]: https://www.chezmoi.io/

I ran the bootstrap command — `chezmoi init ltiao` — got prompted for some
config, answered yes to apply, and then... nothing visibly changed. `chezmoi
status` was empty. The new shell theme wasn't applied. I shrugged, assumed I'd
misunderstood something, and moved on.

What had actually happened: chezmoi's default behavior is to clone the remote
repo's default branch, which on my repo was still `master` — the flat
pre-chezmoi history. The clone succeeded; there was just nothing template-shaped
inside it. chezmoi had nothing to apply.

The wrong-branch clone sat there harmlessly while my old bare-repo setup kept
working. When I came back to figure out what had gone wrong, the answer turned
out to be that single missing flag. The lesson — and the punchline of this
whole post — is that getting your tooling right is harder than the tooling
itself. The recent revamp was the third try, and the one that finally stuck.

## Era 3: The setup today

The current setup is one chezmoi repo at
[`github.com/ltiao/dotfiles`](https://github.com/ltiao/dotfiles), branch `main`,
that bootstraps any of my machines with one command:

```bash
chezmoi init --apply ltiao
```

The repo's `main` is now the default branch on GitHub, so the `--branch=main`
flag from the README is no longer necessary (and Era 2 becomes impossible to
repeat).

![TODO — hero shot of the full setup: Ghostty with Catppuccin Mocha, fastfetch banner, and a ready powerlevel10k prompt](hero-shot.png)

The rest of this section walks through what the setup actually consists of,
layer by layer.

### Terminal: Ghostty

I moved off kitty about a year ago. [Ghostty][ghostty] feels like the modern
default — fast, sensible config language, Catppuccin support out of the box,
splits, and per-platform niceties like macOS option-as-alt and translucent
backgrounds. The config is short:

```
theme = Catppuccin Mocha

font-family = JetBrains Mono
{{- if eq .environment "linux-desktop" }}
font-size = 11
{{- else }}
font-size = 14
{{- end }}

cursor-style = block
shell-integration = detect
window-save-state = always
window-padding-x = 4
window-padding-y = 4
unfocused-split-opacity = 0.7

{{- if eq .chezmoi.os "darwin" }}
macos-option-as-alt = true
macos-titlebar-style = tabs
background-opacity = 0.92
background-blur-radius = 20
{{- end }}
```

[ghostty]: https://ghostty.org/

That `{{- if ... }}` syntax is Go template — chezmoi processes any source file
ending in `.tmpl` through the Go template engine before writing it out, with
access to a small `data` namespace populated from a config file. Here, my Linux
laptop and macOS get different font sizes (Linux's font rendering is heavier, so
14pt looks oversized) and macOS gets a translucent titlebar that Linux doesn't
need.

### Shell: zsh + oh-my-zsh + powerlevel10k

zsh as the shell, oh-my-zsh as a familiar plugin manager, [powerlevel10k][p10k]
as the prompt. p10k has the right combination of fast-by-default, customizable,
and not-actively-hostile-to-readers. The instant-prompt feature in particular is
clever: it caches a snapshot of the prompt and renders that immediately on
shell startup, while the actual init runs in the background. The visible result
is that even a heavyweight `.zshrc` produces a prompt within a few milliseconds.

The snippet that enables it lives near the top of `.zshrc`, before any other
init:

```zsh
if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
  source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
fi
```

p10k offers to add that line automatically on first run, but I bake it into the
template so new machines have it from the start and the wizard never has to
prompt.

![TODO — close-up of the powerlevel10k prompt across two lines, ideally showing the git status segment with a dirty working tree](p10k-prompt.png)

[p10k]: https://github.com/romkatv/powerlevel10k

The OMZ plugin list is short and deliberate — `git`, `docker`, `z`,
`colored-man-pages`, `zsh-autosuggestions`, `zsh-syntax-highlighting`, plus a
handful of "noun-helper" plugins for things I use a lot: `gh`, `direnv`, etc.
Extra zsh files live in `~/.oh-my-zsh/custom/` — one per concern (history and
aliases, Claude launchers, etc.), each short enough to hold in my head.

### Multiplexer: tmux, but only on remote servers

I don't use tmux on my local machines — I have splits and tabs in Ghostty,
which covers the same ground without the wrapper layer. On remote servers
it's a different story. SSH sessions die, and any long-running process dies
with them; tmux solves that.

The repo deploys `.tmux.conf` only on remote machines. Locally, `chezmoi apply`
correctly skips it via a single line in `.chezmoiignore`:

```
{{- if or (eq .environment "mac") (eq .environment "linux-desktop") }}
.tmux.conf
{{- end }}
```

This is the pattern for everything env-specific: list it under a conditional in
`.chezmoiignore` and chezmoi never deploys it on the wrong machine.

### Dotfiles management: chezmoi

The chezmoi mental model, once it clicks, is small. You have a source directory
(default `~/.local/share/chezmoi/`) which is a git repo. Files in there with a
`dot_` prefix get the `dot_` replaced with `.` and the file written to `$HOME`.
Files ending in `.tmpl` get rendered through Go templates first. Files named
`run_onchange_*.sh.tmpl` are scripts that re-run any time their content
changes. `.chezmoiignore` selects which files to skip, with template support.
That's about 80% of what you need.

Day-to-day editing goes through chezmoi rather than the destination file:

```bash
chezmoi edit ~/.zshrc         # opens dot_zshrc.tmpl in $EDITOR
chezmoi apply                 # renders it back to ~/.zshrc
chezmoi diff                  # preview before applying
chezmoi update                # git pull + chezmoi apply in one
```

The discipline is that you treat the source as authoritative and never edit
`~/.zshrc` directly — otherwise you create drift between the two, and chezmoi
will eventually prompt you about it. It's a small habit shift; the payoff is
that `chezmoi update` is the entire sync workflow on every other machine.

### Multi-env model

Three environments matter for my dotfiles:

| Environment | Package manager | Git auth |
|---|---|---|
| macOS laptop | Homebrew | macOS Keychain |
| Arch desktop / laptop | pacman | `gh auth git-credential` |
| Remote work servers | dnf | (handled by the platform) |

The chezmoi templates branch on two data variables, `work` (a boolean
distinguishing work-vs-personal machines) and `environment` (a string
identifying which of the three contexts), populated when I first ran `chezmoi
init` on each machine. The
bootstrap install script picks the right package manager:

```bash
_pkg_install() {
{{ if eq .chezmoi.os "darwin" -}}
  brew install "$@"
{{ else if eq .environment "linux-desktop" -}}
  sudo pacman -S --noconfirm "$@"
{{ else -}}
  sudo dnf install -y "$@"
{{ end -}}
}
```

The gitconfig template branches similarly to pull in `gh auth git-credential`
helpers on the Linux desktop, where I use `gh` for GitHub auth — but not on
macOS (Keychain handles it) or on work servers (different auth model entirely).

This is the part that took the longest to design right. The temptation when
your config diverges across machines is to maintain three branches, or three
repos, or three files with different names. chezmoi's templating lets you keep
one repo and express divergence as conditionals, which means changes propagate
to all machines unless you actively gate them. That asymmetry — share by
default, diverge by exception — matches how I actually want to think about it.

### Tool layer: fzf, fastfetch, uv, Claude launchers

The shell config wires up a handful of tools that aren't shell themselves but
shape day-to-day usage:

- **[fzf][fzf]** for fuzzy file/command/history search. Bound to the usual
  `Ctrl-R`, `Ctrl-T`, `Alt-C` keys via the oh-my-zsh `fzf` plugin.
- **[fastfetch][fastfetch]** prints system info on shell open. (neofetch's
  unmaintained-and-archived state finally pushed me off it.) Worth noting:
  fastfetch has to run *before* the p10k instant-prompt block in `.zshrc`,
  because output during the captured-init region triggers a noisy warning. The
  fix is a one-liner above the preamble, which I'll come back to in the next
  section.

  ![TODO — fastfetch output on shell open, showing OS / kernel / shell / DE / CPU / GPU / memory block alongside the distro ASCII logo](fastfetch.png)
- **[gh][gh]** for everything GitHub — auth, PR review, repo edits. Setting it
  as the git credential helper via `gh auth setup-git` is what makes
  `git push` from chezmoi-managed repos work without a separate password
  prompt.
- **[direnv][direnv]** for per-project environment loading.
- **[uv][uv]** for Python. Single Rust binary that replaces pyenv, virtualenv,
  virtualenvwrapper, pip, and pipx — all in one. Faster than any of them, and
  importantly, doesn't need any shell-init dance: there's no
  `eval "$(uv init)"` equivalent, so the whole "pyenv shims aren't on PATH"
  class of problems just doesn't exist.

[fzf]: https://github.com/junegunn/fzf
[fastfetch]: https://github.com/fastfetch-cli/fastfetch
[gh]: https://cli.github.com/
[direnv]: https://direnv.net/
[uv]: https://github.com/astral-sh/uv

The one I'll spend a paragraph on is the **Claude Code launcher** — a small
set of aliases I use heavily and which is the best example I have of why
multi-env templating earns its keep.

I want one command, `cc`, that starts Claude Code with whatever model and
context settings I prefer. On my local machines, that's just a shell function
that calls `claude` with some flags. On a remote server, I want the same
command to wrap the session in tmux so that an SSH disconnect doesn't kill it
— and I want to set `CLAUDE_CODE_NO_FLICKER=1` because Claude's default redraws
otherwise confuse tmux's scrollback. Same alias, different behavior, expressed
once in a template:

```zsh
_cc() {
  local session="${1:-claude-$(date +%s)}"
  shift 2>/dev/null
{{- if and .work (ne .environment "mac") }}
  export CLAUDE_CODE_NO_FLICKER=1
  if [ -n "$TMUX" ]; then
    command claude "$@"
  elif tmux has-session -t "$session" 2>/dev/null; then
    ta "$session"
  else
    tmux new-session -s "$session" -d "CLAUDE_CODE_NO_FLICKER=1 claude ${(j: :)${(q)@}}" \; attach-session -t "$session"
  fi
{{- else }}
  command claude "$@"
{{- end }}
}

cc()  { _cc "${1:-claude-$(date +%s)}" --model 'claude-opus-4-6[1m]' --effort max "${@:2}"; }
cch() { _cc "${1:-claude-$(date +%s)}" --model 'claude-opus-4-6[1m]' --effort high "${@:2}"; }
cc7() { _cc "${1:-claude-$(date +%s)}" --model 'claude-opus-4-7[1m]' --effort max "${@:2}"; }
ccs() { _cc "${1:-claude-$(date +%s)}" --model 'claude-sonnet-4-6[1m]' --effort max "${@:2}"; }
# ... more variants for different model/context/effort combinations
```

Same muscle memory across every machine, with the wrapping that each
environment needs added invisibly. A `cchelp` function dumps the full list of
variants for when I forget which suffix means what.

## The hard parts

Not everything was smooth. Three problems I hit during this rewrite are worth
sharing because they're easy traps and the symptoms don't always point at the
real cause.

**1. `.chezmoiignore` patterns must use destination paths, not source paths.**
This bit me hardest. chezmoi's source files are named `dot_zshrc.tmpl`,
`dot_tmux.conf`, etc. — `dot_` prefix, optional `.tmpl` suffix. When you list
patterns in `.chezmoiignore`, it's natural to write `dot_tmux.conf`. That's
*wrong*. chezmoi expects destination paths there — `.tmux.conf`. The kicker:
chezmoi accepts the wrong form silently. Patterns using source paths match
nothing and are no-ops. I had a `.chezmoiignore` rule that was supposed to skip
`.tmux.conf` on local machines for two years, and it had been doing nothing the
whole time. Other rules in the same file appeared to work, but only because the
files they pointed at rendered empty under their internal template guards. The
fix is a one-character search-and-replace, but finding the bug needed a careful
read of chezmoi's docs and a moment of realization while staring at the wrong
file showing up in `$HOME`.

**2. Oh-my-zsh plugin load order vs. your custom init.** I had a custom
`~/.oh-my-zsh/custom/pyenv.zsh` that ran `pyenv init -`, plus the oh-my-zsh
`pyenv` plugin in my plugin list. Every shell startup printed `Found pyenv, but
it is badly configured (missing pyenv shims in $PATH)`. I assumed the warning
was from pyenv itself, spent some time chasing PATH ordering, and only realized
much later that the warning is emitted by the OMZ pyenv plugin — which loads
during `source $ZSH/oh-my-zsh.sh`, *before* `~/.oh-my-zsh/custom/*.zsh` files
get sourced. My custom init was running too late. The fix is either to move
pyenv init above `plugins=()`, or drop the redundant OMZ plugin and rely on the
custom file. I went with the third option: switch to uv and delete all of it.

![TODO — screenshot of the "Console output during zsh initialization detected" warning from p10k, with the captured-output region visible below the "─────" line](p10k-instant-prompt-warning.png)

**3. p10k's instant-prompt and the "console output during init" warning.**
Instant-prompt's contract is that nothing should write to the console between
the preamble and the first prompt — otherwise p10k can't cleanly buffer and
flush the output, and you get an angry warning every time you open a shell.
fastfetch prints to the console by definition, so I had a problem. My first
attempt was a deferred `precmd` hook — register a one-shot callback that calls
fastfetch right before the first prompt — which seemed clean but still
triggered the warning, because p10k's own finalize hook is registered *after*
mine (it lives in `~/.p10k.zsh`, which is sourced last). The fix that actually
worked is the dumbest one: just move the `fastfetch` call *before* the
instant-prompt preamble in `.zshrc`. Now it runs before init starts, gets
written directly to the terminal, and the prompt renders cleanly under it.

## Closing thoughts

The two metrics I judge a dotfiles setup by:

1. How long does it take to bootstrap a fresh machine to a working state?
2. How long does it take to propagate a change to all my machines?

For (1) the answer is now `chezmoi init --apply ltiao` plus the time pacman or
brew takes to install five extra packages. Total wall-clock under five minutes,
most of which is network. For (2) it's `chezmoi edit <file>; chezmoi apply` on
one machine, `git commit && git push`, then `chezmoi update` on each of the
others — a workflow short enough to be cheap.

It took three takes to get here, and the third take wasn't smooth either —
this post itself is partly a record of the bugs I found while writing it. But
the time invested in tooling compounds. A couple of hours of fiddly migration
buys years of fast machine setups. That trade is one of the best I make at this
point in my career.

Up next: nothing urgent. I might write a follow-up on the Claude Code launcher
itself — the version above is doing real work, and the env-conditional model
generalizes to anything where the same alias should behave differently in
different contexts. If you'd like to compare notes on your own setup, the repo
is at [`github.com/ltiao/dotfiles`](https://github.com/ltiao/dotfiles).
