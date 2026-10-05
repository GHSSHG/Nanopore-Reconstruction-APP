"""Network pieces that can be checked on the CPU without the released weights."""

from __future__ import annotations

import numpy as np


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def reference_lstm(x, x_kernel, x_bias, h_kernel, forget_bias):
    """Plain LSTM over x (B, T, D), one step at a time; gate order i, f, g, o."""
    batch, steps, _ = x.shape
    hidden = h_kernel.shape[0]
    h = np.zeros((batch, hidden))
    c = np.zeros((batch, hidden))
    out = np.empty((batch, steps, hidden))
    for t in range(steps):
        i, f, g, o = np.split(x[:, t] @ x_kernel + x_bias + h @ h_kernel, 4, axis=-1)
        c = sigmoid(f + forget_bias) * c + sigmoid(i) * np.tanh(g)
        h = sigmoid(o) * np.tanh(c)
        out[:, t] = h
    return out


def test_bilstm_directions_share_one_scan():
    import jax

    from nanorecon.runtime.network import LSTM_UNROLL, _bilstm

    rng = np.random.default_rng(0)
    batch, steps, dim, hidden = 2, LSTM_UNROLL + 3, 5, 4  # a partial unrolled iteration at the end
    x = rng.standard_normal((batch, steps, dim)).astype(np.float32)
    p = {
        f"{d}_{name}": (rng.standard_normal(shape) * 0.5).astype(np.float32)
        for d in ("fwd", "bwd")
        for name, shape in (("x_kernel", (dim, 4 * hidden)), ("x_bias", (4 * hidden,)), ("h_kernel", (hidden, 4 * hidden)))
    }
    with jax.default_matmul_precision("highest"):  # on a GPU the default would be TF32
        fwd, bwd = _bilstm(x, p, hidden, 1.0)
    expected_fwd = reference_lstm(x, p["fwd_x_kernel"], p["fwd_x_bias"], p["fwd_h_kernel"], 1.0)
    expected_bwd = reference_lstm(x[:, ::-1], p["bwd_x_kernel"], p["bwd_x_bias"], p["bwd_h_kernel"], 1.0)[:, ::-1]
    np.testing.assert_allclose(np.asarray(fwd), expected_fwd, atol=1e-5)
    np.testing.assert_allclose(np.asarray(bwd), expected_bwd, atol=1e-5)


def tf32_rna(x):
    """fp32 rounded to TF32's 10 mantissa bits, nearest with ties away from zero (NumPy)."""
    bits = np.asarray(x, np.float32).view(np.uint32)
    return ((bits + np.uint32(0x1000)) & np.uint32(0xFFFFE000)).view(np.float32)


def test_to_f16_rounds_like_tf32():
    from nanorecon.runtime.network import to_f16

    ulp = 2.0**-10
    x = np.array([1 + 0.75 * ulp, 1 + 0.5 * ulp, 1 + 0.25 * ulp, -(1 + 0.5 * ulp), 3.0e-3, -12345.678], np.float32)
    got = np.asarray(to_f16(x)).astype(np.float32)
    np.testing.assert_array_equal(got, tf32_rna(x))
    assert got[1] == 1 + ulp and got[3] == -(1 + ulp)  # ties away from zero


def test_pow2_scale_is_exact_and_leaves_headroom():
    from nanorecon.runtime.network import pow2_scale

    for bound in (1e-6, 0.7, 1.0, 27.1, 1000.0, 40000.0):
        s = float(pow2_scale(np.float32(bound)))
        assert np.log2(s) == int(np.log2(s)) and 2.0**14 <= bound * s < 2.0**15


def test_search_finds_the_nearest_codeword_and_lowest_index_on_ties():
    from nanorecon.runtime.network import nearest_codeword, search_codebook

    rng = np.random.default_rng(3)
    codebook = rng.standard_normal((1000, 16)).astype(np.float32)
    codebook[[9, 700]] = codebook[5]  # duplicates in the same and a later tile: ties go to the lowest index
    z = (codebook[rng.integers(0, 1000, 300)] + 0.3 * rng.standard_normal((300, 16))).astype(np.float32)
    z[0] = codebook[5]
    got = np.asarray(nearest_codeword(z, search_codebook(codebook)))  # 4 tiles of 256, the last one padded
    d = ((z[:, None, :].astype(np.float64) - codebook[None, :, :]) ** 2).sum(-1)
    best = d.argmin(1)
    assert got[0] == 5
    chosen, exact = d[np.arange(300), got], d[np.arange(300), best]
    assert (got == best).mean() >= 0.99 and np.all(chosen <= exact * (1 + 1e-3) + 1e-6)  # only near-ties may differ


def test_f16_dense_equals_tf32_inputs_with_fp32_sums():
    import jax

    from nanorecon.runtime.network import F16Dense

    rng = np.random.default_rng(4)
    x = (rng.standard_normal((3, 5, 64)) * np.exp(rng.uniform(-12, 3, (3, 5, 64)))).astype(np.float32)  # tiny to large
    kernel = (rng.standard_normal((64, 8)) * 0.05).astype(np.float32)
    bias = rng.standard_normal(8).astype(np.float32)
    layer = F16Dense(8)
    got = np.asarray(layer.apply({"params": {"kernel": kernel, "bias": bias}}, x, np.abs(x).max()))
    expected = tf32_rna(x).astype(np.float64) @ tf32_rna(kernel).astype(np.float64) + bias
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
    assert jax.eval_shape(layer.init, jax.random.PRNGKey(0), x, 1.0)["params"]["kernel"].shape == (64, 8)  # nn.Dense layout
