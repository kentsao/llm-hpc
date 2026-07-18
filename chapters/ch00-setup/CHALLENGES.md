# ch00 · Challenges

Commit evidence (numbers, plots, short write-ups) to `chapters/ch00-setup/results/`.

### ★ C0.1 — The workflow rehearsal
Create a branch `ch00-setup`, complete the exercise, open a PR, watch CI run,
merge it. Then tick the boxes in `PROGRESS.md` in a follow-up commit.
*Skill examined: the git/PR/CI loop you'll use for every chapter.*

### ★★ C0.2 — Beat your own triad
Implement two more triad variants and benchmark all three on ≥400 MB of arrays:
1. the naive `a[:] = b + scalar * c` (allocates temporaries),
2. a temporary-free version (`np.multiply(c, scalar, out=a); a += b`),
3. a `numba` `@njit(parallel=True)` version.

Report GB/s for each and the % of your machine's spec-sheet bandwidth. Explain
*why* the temporary-free version is faster — count the actual bytes each moves.
*Skill examined: reasoning about memory traffic, not just timing it.*

### ★★ C0.3 — First contact with a cloud GPU
On a free Colab T4: clone your repo, run the triad with CuPy or PyTorch on the GPU,
and measure the T4's memory bandwidth (spec: 320 GB/s). Save the notebook to
`chapters/ch00-setup/results/` with your achieved number.
*Skill examined: the Colab round-trip workflow used in ch03–ch07.*

### ★★★ C0.4 — How trustworthy is a timer?
Design an experiment measuring how benchmark noise responds to conditions:
run the triad 100× under (a) idle machine, (b) heavy background load (e.g., a
video call or `yes > /dev/null` × 4), (c) low-power mode if available. Plot the
timing distributions (histogram or violin). Report median, mean, and IQR for each,
and write 3–5 sentences: which statistic was stable, which lied, and what you'll
do about it for the rest of the curriculum.
*Skill examined: experimental methodology — the difference between a number and a measurement.*
