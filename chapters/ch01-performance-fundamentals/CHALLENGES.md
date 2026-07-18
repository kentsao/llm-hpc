# ch01 · Challenges

Commit evidence to `chapters/ch01-performance-fundamentals/results/`.

### ★ C1.1 — How fast is your BLAS?
Benchmark `float32` NumPy matmul at n = 256, 512, 1024, 2048, 4096. Plot achieved
GFLOP/s vs n. At what size does throughput saturate, and what fraction of the
compute ceiling you measured in `examples/roofline.py` does it reach? Explain the
small-n behavior using arithmetic intensity.
*Examined: connecting measurements to the roofline model.*

### ★★ C1.2 — Roofline your own kernels
Add four kernels of your choice to the roofline plot (ideas: `np.exp`, prefix sum,
`x @ w + b` fused vs unfused, float64 vs float32 matmul). For each, *predict* its
position (compute the AI by hand) before measuring, and mark predictions vs
measurements on the plot. Report where your prediction was most wrong and why.
*Examined: cost-model reasoning — predict first, measure second.*

### ★★ C1.3 — Fix the slow script
`examples/profile_demo.py` is dominated by its input pipeline. Optimize
`load_batch` (vectorize it) until the *matmul* is the top profile entry, and get a
total step speedup ≥ 10×. Use Amdahl to show your measured total speedup is
consistent with the fraction you optimized.
*Examined: the profile → predict → fix → re-measure loop.*

### ★★★ C1.4 — An analytic cost model for a transformer layer
On paper (markdown file), derive FLOPs and minimum bytes per token for one GPT-2-small
transformer layer (d_model=768, 12 heads, seq=1024, batch=1): QKV projections,
attention scores, attention-weighted values, output projection, MLP (4× expansion).
Which sub-op has the lowest AI? Predict, for the T4's roofline, which ops are
memory- vs compute-bound. Keep this file — ch06 profiles the real thing and you'll
grade your own predictions.
*Examined: the exact analysis that motivates FlashAttention and kernel fusion.*
