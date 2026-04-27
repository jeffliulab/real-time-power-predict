# CLAUDE.md

Repo-specific operating rules for Claude. Loaded automatically into context.

---

## ⛔⛔⛔ ABSOLUTE PROHIBITION — NO LOOPS, NO TIMERS, NO SCHEDULED TASKS ⛔⛔⛔

**THIS IS THE FIRST RULE OF THIS REPO. READ EVERY SESSION. NO EXCEPTIONS.**

**Why this rule exists**: On 2026-04-27 a forgotten `until ssh tufts-login ...; do
sleep 30; done` loop ran for 4 days, fired ~14,000 DUO push attempts, and
**locked the user's Tufts DUO account**. Re-violation could permanently disable
their academic account, lose their HPC access, and end their CS 137 work.

### The rule, simply stated

> **DO NOT** create any task, command, script, tool call, or background process
> that runs more than once on any kind of timer, loop, schedule, or condition.
>
> **NO** loops that wait. **NO** timers. **NO** scheduled wakeups. **NO** periodic
> polling. **NO** auto-retry. **NO** auto-resume. **NO** background daemons.
>
> Every action you take must be **single-shot** — execute once, return result,
> done. If the user wants something checked again, they will explicitly tell
> you to check it again.

### Forbidden — Shell patterns

```bash
# === LOOP PATTERNS ===
until <cond>; do ...; done                # forbidden
while <cond>; do ...; done                # forbidden
while true; do ...; done                  # forbidden
while :; do ...; done                     # forbidden
for i in $(seq ...); do sleep N; done     # forbidden if N > 0 inside
for ((;;)); do ...; done                  # forbidden
repeat N do ...                           # forbidden

# === SLEEP / WAIT PATTERNS ===
sleep N && <command>                      # forbidden (any wait-then-act)
sleep N; <command>                        # forbidden (any wait-then-act)
wait $!; <command>                        # forbidden
sleep N (in any background context)       # forbidden if > 5 seconds

# === BACKGROUND / DETACH PATTERNS ===
<command> &                               # forbidden if command is non-local
nohup <command>                           # forbidden in all cases
disown                                    # forbidden in all cases
setsid                                    # forbidden in all cases
tmux new-session -d                       # forbidden — detached tmux
screen -dmS                               # forbidden — detached screen

# === FILE-WATCH / MONITOR PATTERNS ===
tail -f <log>                             # forbidden — blocks indefinitely
watch <command>                           # forbidden — schedules repeated runs
inotifywait                               # forbidden
fswatch                                   # forbidden
entr                                      # forbidden
nodemon                                   # forbidden

# === SCHEDULED TASK PATTERNS ===
crontab -e / crontab <file>               # forbidden — never modify user's crontab
at <time> <command>                       # forbidden — no `at` jobs
launchctl bootstrap / launchctl load      # forbidden — no new launchd jobs
launchd <plist>                           # forbidden — no new daemons
systemctl --user enable                   # forbidden — no systemd units

# === RETRY / AUTO-RECONNECT PATTERNS ===
autossh                                   # forbidden — auto-reconnects ssh
mosh <host>                               # forbidden — auto-reconnects
ssh -o ServerAliveInterval=N (high freq)  # forbidden if frequent enough to keepalive
docker run --restart                      # forbidden — auto-restart
vagrant up (with autostart)               # forbidden
```

### Forbidden — Python patterns

```python
# === INFINITE LOOPS ===
while True: ...                           # forbidden if any I/O
while not <cond>: time.sleep(N); ...      # forbidden
itertools.count(...)                      # forbidden as loop counter
itertools.cycle(...)                      # forbidden

# === SLEEP / WAIT ===
time.sleep(N) (in any non-tiny context)   # forbidden if N > 5 or in loop
asyncio.sleep(N) in async loop            # forbidden

# === RETRY DECORATORS ===
@tenacity.retry(...)                      # forbidden
@backoff.on_exception(...)                # forbidden
@retrying.retry(...)                      # forbidden
for attempt in range(N): ...except:retry  # forbidden retry loop

# === ASYNC / THREAD / MULTIPROC ===
asyncio.run + sleep loop                  # forbidden
loop.run_forever()                        # forbidden
threading.Thread(target=poll_fn, ...)     # forbidden if poll_fn loops
multiprocessing.Process for polling       # forbidden
concurrent.futures with continuous tasks  # forbidden

# === SCHEDULER LIBRARIES ===
apscheduler.BackgroundScheduler()         # forbidden — periodic tasks
schedule.every(N).seconds.do(...)         # forbidden
celery beat / celery worker               # forbidden
rq-scheduler                              # forbidden

# === SSH CLIENT AUTOMATION ===
paramiko (if any auto-reconnect or loop)  # forbidden
fabric (if any auto-retry)                # forbidden
pexpect (interactive sessions)            # forbidden if ssh involved
```

### Forbidden — Claude-runtime tools

The following Claude tools are **FORBIDDEN** in this repo for any use that
involves Tufts HPC, ssh, or auto-repeat:

- **`Monitor`** — Forbidden if the monitored process touches Tufts HPC at all.
  Even monitoring is a periodic operation; the rule is single-shot.
- **`ScheduleWakeup`** — Forbidden in this repo. We do NOT schedule "check
  back later" wakeups. The user will tell us when to check again.
- **`CronCreate`** / **`CronList`** / **`CronDelete`** — Forbidden. Never
  install any cron-style scheduled task.
- **`/loop` skill** — Forbidden in this repo for any task that touches
  Tufts HPC. Single-shot only.
- **`Bash` with `run_in_background: true` containing a loop** — Forbidden.
  Use `run_in_background` ONLY for genuinely one-shot long commands (e.g. a
  single `git push` of a large file) that will exit on their own without any
  loop, sleep, or wait pattern inside.

### Why so strict

Tufts DUO triggers per SSH channel. **One ssh = one DUO push.** A loop that
fires `ssh tufts-login` every N seconds = `60/N` DUO pushes per minute. Even
at 1 push every 5 minutes, that's 288/day. Tufts IT auto-locks accounts at
roughly 50–100 unanswered DUO pushes per day.

The 2026-04-27 incident: 30-second polling loop × 4 days = ~14,000 DUO push
attempts. Account locked. Recovery required IT ticket + manual unlock.

**Do not assume "just one short loop is fine."** It is not. The user has
explicitly said this rule has zero tolerance.

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

There is exactly **ONE** approved way to wait for HPC work to finish:

- **The user tells you when to check.** They start a SLURM job, walk away, and
  come back to you later with "check it now" or "is it done?". You then do
  one bundled ssh, report, done. The wait happens **outside this conversation,
  in the user's head/clock**, not in any code/timer/loop you control.

Server-side helpers are also approved (they don't run locally):

- **SLURM email notification** (`#SBATCH --mail-type=END --mail-user=...`)
  — Tufts HPC sends the email when the job ends. Costs nothing on this Mac.
- **SLURM job dependency chains** (`sbatch --dependency=afterany:JOBID ...`)
  — One ssh creates a chain of jobs that the HPC scheduler runs autonomously.
  No local code monitors anything.

Forbidden, even though they sound innocent:

- ❌ "I'll just check back in 5 minutes" via `ScheduleWakeup`
- ❌ "Let me poll once every minute for 10 minutes" via a bounded loop
- ❌ "Tail the SLURM log so we see new lines" via `tail -f` over ssh
- ❌ "Watch for the file to appear" via `until [ -f X ]; do sleep; done`
- ❌ "Auto-resubmit if the job fails" via any retry pattern

If any of those are tempting, the answer is: **STOP. Ask the user. They will
say when to check.**

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
