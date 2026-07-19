# Result
I got three result. Actual byte move I think is 3 * n * 8 (float64) -> n is the number of element, in this exercise, it has been set to 50,000,000. One movement is read b, c each for 8 bytes and write a which is 8 bytes operation too.
## Naive
STREAM triad bandwidth: 21.8 GB/s -> about 1/3 of SPEC

## Numpy Multiply
STREAM triad bandwidth: 30.0 GB/s -> about 1/2 of SPEC

## numba njit with naive
STREAM triad bandwidth: 47.0 GB/s -> about 2/3 of SPEC

# Why temporary-free version is faster
Temporary-free version is faster because that no extra data movement and can reduce the overall time of computation.

