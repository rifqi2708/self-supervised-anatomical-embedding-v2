# Quadra branch workflow

Use `uae-quadra-validation` as the shared integration branch for new Quadra work.
Keep `main` as the existing default/reference branch and `dev` temporarily until
its operational role is reviewed. Changing the default branch is a separate
project decision.

## Start and finish a task

1. Check `git status` and preserve unfinished work before switching branches.
2. Fetch the current integration branch. Start a `codex/<task>` branch from it,
   using an isolated checkout when the current checkout contains unfinished work.
3. Keep each task focused. Target its pull request to `uae-quadra-validation`.
4. Review the change and run checks appropriate to the affected behavior.
5. Integrate the reviewed work. Prefer history-preserving merges when they help
   trace research development.
6. Confirm the task commits are reachable from the remote integration branch,
   preserve any uncommitted work, and resolve dependent pull requests before
   retiring the task branch locally and remotely. Squashed or cherry-picked work
   requires a separate content-equivalence check.

Branch count should reflect active work. Keep only task branches with a current
purpose; preserve historical development through retained commits and release
tags rather than long-lived completed task branches.

## Reproducibility and recovery

Record exact commit IDs and relevant immutable release tags in experiment
manifests. Branch names can move and are insufficient to identify executed code.
The frozen `quadra-disposable-v1` tag must remain unchanged; integration changes
do not automatically change the tagged production workflow.

A Git commit or bundle preserves committed source history. It does not preserve
unfinished files, datasets, masks, models, environments or generated evidence.
Follow `AGENTS.md` and `environment/ARTIFACT_BACKUP.md` for artifact routing and
backup. Review content before pushing to this public repository.

## Cleanup on 30 September 2026

Ten remote task branches and eleven local task branches were found to have no
commits outside the existing Quadra integration baseline. Draft PRs #1 and #2
were closed as superseded, with their original records preserved locally.
The flagged-mask and SuperPoint work were integrated separately after resolving
conflicts and checking affected behavior. The flagged-mask branch remains while
its primary checkout contains unfinished research work. Both existing dirty
checkouts were preserved; missing temporary worktree registrations were pruned.

The detailed branch-to-commit mapping, recovery bundle, unfinished-file archives,
checksums and execution records are in the local archive under
`metadata/manifests/repository-branch-cleanup-20260930T083129Z/`.
