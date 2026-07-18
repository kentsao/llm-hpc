"""Exercise 2: FLOPs, bytes, arithmetic intensity — the accounting behind the roofline.

Fill in the cost models, then classify kernels on given hardware. Assume:
- dtype is float32 (4 bytes) unless stated,
- "bytes" = MINIMUM traffic: each input read once from memory, each output
  written once (perfect caching of intermediates).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Hardware:
    peak_gflops: float   # compute ceiling, GFLOP/s
    peak_gbs: float      # memory bandwidth ceiling, GB/s

    @property
    def ridge_point(self) -> float:
        """AI (FLOP/byte) where the two ceilings meet."""
        return self.peak_gflops / self.peak_gbs


M1_CPU = Hardware(peak_gflops=100.0, peak_gbs=68.0)
T4_GPU = Hardware(peak_gflops=8100.0, peak_gbs=320.0)


def elementwise_cost(n: int, flops_per_elt: int = 1, n_inputs: int = 2) -> tuple[int, int]:
    """(flops, bytes) for an elementwise op over n float32 elements with
    `n_inputs` input arrays and one output array."""
    return n * flops_per_elt, (n_inputs + 1) * n * 4


def matmul_cost(m: int, k: int, n: int) -> tuple[int, int]:
    """(flops, bytes) for (m,k) @ (k,n) in float32, minimum-traffic convention."""
    return 2 * m * k * n, (m * k + k * n + m * n) * 4


def attention_scores_cost(batch: int, heads: int, seq: int, head_dim: int) -> tuple[int, int]:
    """(flops, bytes) for computing S = Q @ K^T over all batch*heads:
    Q, K are (batch, heads, seq, head_dim); S is (batch, heads, seq, seq), float32.

    This is the matrix that FlashAttention famously avoids writing — computing
    its size here is the setup for ch07.
    """
    flops = 2 * batch * heads * seq * seq * head_dim
    nbytes = (2 * batch * heads * seq * head_dim + batch * heads * seq * seq) * 4
    return flops, nbytes


def arithmetic_intensity(flops: int, nbytes: int) -> float:
    return flops / nbytes


def classify(flops: int, nbytes: int, hw: Hardware) -> str:
    """Return "memory-bound" or "compute-bound" for this kernel on this hardware."""
    if arithmetic_intensity(flops, nbytes) < hw.ridge_point:
        return "memory-bound"
    return "compute-bound"


def predicted_gflops(flops: int, nbytes: int, hw: Hardware) -> float:
    """Roofline prediction: best-achievable GFLOP/s for this kernel on hw."""
    return min(hw.peak_gflops, arithmetic_intensity(flops, nbytes) * hw.peak_gbs)


if __name__ == "__main__":
    for name, (f, b) in {
        "GeLU 16M elts": elementwise_cost(16_000_000, flops_per_elt=8, n_inputs=1),
        "matmul 4096^3": matmul_cost(4096, 4096, 4096),
        "attn scores (1,12,2048,64)": attention_scores_cost(1, 12, 2048, 64),
    }.items():
        ai = arithmetic_intensity(f, b)
        print(f"{name:<28s} AI={ai:8.2f}  M1:{classify(f, b, M1_CPU):<14s} T4:{classify(f, b, T4_GPU)}")
