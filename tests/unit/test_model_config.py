"""The fixed model description: network interpretation of the release config and the profile."""

from __future__ import annotations

import json

import numpy as np
import pytest

from nanorecon.model_config import MODEL, PROFILE, CodecProfile, parse_network_config

from ..helpers import RELEASE_CONFIG


def test_network_interpretation_matches_training_factory():
    net = parse_network_config(RELEASE_CONFIG["model"])
    assert net.downsample_factor == 16 and net.tokens_for(8192) == 512
    assert net.attention_backend == "cudnn"
    assert net.residual_correction_hidden_dim == 192 and net.residual_correction_alpha_max == 0.1
    assert net.encoder_use_block_norm and net.encoder_use_input_norm and net.encoder_use_transition_norm
    assert net.diveq_sigma2 == 0.001  # kept as a fact; inference never uses it
    ids = net.band_ids()
    assert len(ids) == 257 and ids[20] == 0 and ids[21] == 1 and ids[51] == 1 and ids[52] == 2 and ids[103] == 3


def test_profile_agrees_with_the_network():
    net = parse_network_config(RELEASE_CONFIG["model"])
    p = PROFILE
    assert p.hop_samples == p.chunk_samples - p.overlap_samples and 0 <= p.overlap_samples < p.chunk_samples
    assert net.tokens_for(p.chunk_samples) == p.tokens_per_chunk
    assert p.tokens_per_chunk * net.istft_hop_length == p.chunk_samples  # decoder output length
    assert p.codebook_size == net.codebook_size == MODEL.codebook_size <= np.iinfo(np.uint16).max + 1
    assert p.sample_rate_hz == net.istft_sample_rate == RELEASE_CONFIG["input_signal"]["sample_rate_hz"]
    assert p.chunk_samples == RELEASE_CONFIG["input_signal"]["segment_samples"]


def test_profile_json_round_trip():
    data = json.loads(json.dumps(PROFILE.to_json_dict()))
    assert CodecProfile.from_json_dict(data) == PROFILE
    with pytest.raises(ValueError, match="exactly the fields"):
        CodecProfile.from_json_dict({**data, "extra": 1})
    del data["stitch"]
    with pytest.raises(ValueError, match="exactly the fields"):
        CodecProfile.from_json_dict(data)
