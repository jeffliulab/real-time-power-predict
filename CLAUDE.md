# CLAUDE.md

Repo-specific operating rules for Claude. Loaded automatically into context.

---

## ⛔ ZERO LOOP / ZERO WAIT — ABSOLUTE PROHIBITION

**Read this section FIRST every session. Violation locked the user's Tufts DUO
account on 2026-04-27 with ~14,000 background DUO push attempts over 4 days.
Re-violation could permanently disable their academic account.**

### Forbidden patterns (under any circumstance, in any tool)

The following patterns are **STRICTLY FORBIDDEN** in any Bash invocation,
inline shell command, generated script, or background task — regardless of
intent, scope, or duration:

```bash
until <cond>; do ...; done                # forbidden
while <cond>; do ...; done                # forbidden (use bounded for)
while true; do ...; done                  # forbidden
while :; do ...; done                     # forbidden
for i in $(seq ...); do sleep N; done     # forbidden if any network/HPC inside
sleep N && <command>                      # forbidden if N > 5 and command is non-local
sleep N; ssh ...                          # forbidden
ssh ... &                                 # forbidden (background ssh)
nohup ssh ...                             # forbidden
disown                                    # forbidden after any ssh
tail -f                                   # forbidden if it's monitoring an ssh log
watch <command>                           # forbidden in any context
tmux new-session -d                       # forbidden
screen -dmS                               # forbidden
crontab                                   # forbidden — never write to user's crontab
launchctl bootstrap / launchctl load      # forbidden — no new launchd jobs
```

Python equivalents are also forbidden:

```python
while True: ...                           # forbidden if any I/O / network
while not <cond>: time.sleep(N); ...      # forbidden
for attempt in range(N): ssh; sleep       # forbidden retry loop
itertools.cycle(...)                      # forbidden
asyncio.run + asyncio.sleep loop          # forbidden
threading.Thread(target=poll_fn, ...)     # forbidden
multiprocessing.Process for polling       # forbidden
apscheduler / schedule / celery / cron    # forbidden — no schedulers
tenacity / backoff / retrying             # forbidden — no auto-retry libraries
```

**Why so strict:** The user's DUO triggers per SSH channel. A loop that
fires `ssh tufts-login` every N seconds = N DUO pushes per minute. Even at
1 push every 5 minutes, that's 288/day → cluster IT auto-locks at ~50.

### TaskStop is NOT enough

When using `Bash` with `run_in_background: true`, the resulting task ID
**only tracks Claude's view of the process**. The underlying zsh shell
spawned via `eval` is a separate OS process that:
- Has PPID = 1 after detachment
- Survives `TaskStop` calls
- Continues executing whatever inline command/loop until killed at OS level
- Has no automatic timeout

**MANDATORY verification protocol after every `TaskStop`:**

```bash
# 1. List any zsh/bash processes still alive
ps -axwww -o pid,ppid,start,command | grep -E 'zsh -c|bash -c' | grep -v grep

# 2. For each suspect, check command — if it contains ssh/while/until/sleep/tufts:
kill -9 <PID>

# 3. Re-verify dead
ps -p <PID> 2>&1
```

If you discover an orphan shell from a prior session in any startup audit:
**kill it immediately**, before any other work.

### Approved alternatives for "wait until X happens"

- **One-shot user-driven polling**: user explicitly says "go check now" → one
  bundled ssh command → report → done.
- **Email notification from SLURM** (`#SBATCH --mail-type=END --mail-user=...`)
  — server-side, costs nothing locally.
- **`ScheduleWakeup` tool** (Claude-runtime managed) — terminates cleanly at
  conversation end; far safer than Bash background loops.
- **Bounded `for` loops** with explicit iteration cap — `for i in $(seq 1 5); do ...; done`
  with no remote network call inside.

---

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

2. **Do not poll HPC in the background.** Zero exceptions. See the
   **ZERO LOOP** section above for the full prohibition. Even a single
   `until ssh ...; do sleep` invocation can run for days if `TaskStop`
   doesn't kill the underlying shell — verified incident on 2026-04-27.

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
   on next connection. (Mux idle does NOT generate DUO; only new
   channels through it do.)

### When the user says "go" / "check it" / "submit"

Then proceed. Bundle the work into the minimum number of ssh calls.
Mention if a DUO push is expected so the user has their phone ready.

### Cleaning up after work

When the user signals "pause" / "stop" / "I'm stepping away":
- Stop any background poll tasks immediately (`TaskStop`).
- **Verify with `ps -axwww | grep -E 'zsh -c|bash -c'` that no orphan
  shells survived.** If any contain ssh/until/while/sleep/tufts in the
  command, `kill -9 <PID>` them.
- Leave the ssh ControlMaster process alive (per rule 6 above) unless
  user explicitly asks to disconnect.
- Confirm in chat that all HPC access is severed and no orphan shells
  remain.

### Long-running HPC work continues without us

SLURM jobs already running on HPC (training, evaluation) keep going
regardless of local ssh state. We don't need ssh to keep them alive.
Reconnect only to sync results back when the user is ready.

---

## Session-start checklist

At the start of every session in this repo, before doing any work:

```bash
ps -axwww -o pid,start,command | grep -E 'until.*ssh|while.*ssh|sleep.*ssh' | grep -v grep
```

If anything matches → kill it before proceeding. Report to user.

```bash
ps -axwww -o pid,start,command | grep -E 'zsh -c|bash -c' | grep -v 'Local_Root\|grep'
```

Skim for any orphaned background shell with ssh/Tufts/HPC keywords. If
suspicious, kill before proceeding.

This audit takes <5 seconds and prevents catastrophic re-occurrence.
