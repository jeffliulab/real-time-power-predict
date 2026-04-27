# CLAUDE.md

Repo-specific operating rules for Claude. Loaded automatically into context.

## HPC access — strict consent required

Tufts HPC (`tufts-login` / `login.pax.tufts.edu`) requires DUO 2FA at the
PAM **session** stage on every fresh ssh connection. Each unsolicited
ssh/rsync/scp/sbatch attempt fires a DUO push to the user's phone, which
is a real-world disturbance.

### Hard rules

1. **Do not initiate any HPC access without explicit user instruction
   in the current message.** This applies to ssh, rsync, scp, sftp,
   sbatch, squeue, sacct, scancel, and any tool/command that would open
   a network connection to `*.pax.tufts.edu` or `*.tufts.edu`.

2. **Do not poll HPC in the background.** No `until ssh ...; do sleep`
   loops, no scheduled wake-ups that re-check job state. If the user
   wants to monitor a job, do one ssh on user request and report back.

   **CRITICAL** (learned 2026-04-27): Calling `TaskStop` on a Bash
   background task does NOT kill the underlying zsh process if the
   task contained an `eval`-spawned loop. Runaway polling shells
   survive `TaskStop` and run indefinitely. A forgotten `until ssh
   tufts-login ...; do sleep 30; done` ran for 4 days unnoticed and
   locked the user's DUO account with ~14,000 push attempts. **Always
   verify with `ps -axwww | grep -E '<pattern>' | grep -v grep` after
   `TaskStop` and `kill -9 <pid>` if the underlying shell still
   exists.** If you find one of these from a prior session, kill it
   immediately — never leave it running.

3. **Do not request HPC access in non-execution state.** When the user
   has paused, asked you to wait, or stepped away, do nothing on HPC
   even if a previous instruction looks "obviously continuing." Wait for
   a fresh "go" message before reconnecting.

4. **EVERY ssh = 1 DUO push.** Tufts fires DUO at session stage on each
   new SSH channel, so ControlMaster mux does NOT amortize DUO across
   commands (verified 2026-04-27, see memory). Bundle aggressively:
   pack `rsync + sacct + tail + commit + grep` into ONE remote heredoc
   per user request. Never re-ssh to "check just one more thing."

5. **No silent retry on DUO timeout.** A timed-out DUO push means the
   user wasn't ready. Surface the failure, ask the user to confirm
   readiness for another attempt — do NOT auto-retry, every retry =
   another DUO push.

6. **Don't kill the mux master.** Leave any existing
   `cm-pliu07@login.pax.tufts.edu:22` socket and ssh mux process alive
   even when "doing nothing." Killing them forces fresh auth = new DUO
   on next connection.

### When the user says "go" / "check it" / "submit"

Then proceed. Bundle the work into the minimum number of ssh calls.
Mention if a DUO push is expected so the user has their phone ready.

### Cleaning up after work

When the user signals "pause" / "stop" / "I'm stepping away":
- Stop any background poll tasks immediately (`TaskStop`).
- Kill the ssh ControlMaster process and remove the mux socket so the
  user doesn't see lingering connection state.
- Confirm in chat that all HPC access is severed.

### Long-running HPC work continues without us

SLURM jobs already running on HPC (training, evaluation) keep going
regardless of local ssh state. We don't need ssh to keep them alive.
Reconnect only to sync results back when the user is ready.
