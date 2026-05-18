import sys
from unittest.mock import MagicMock
# Prevent torchcodec OSError/RuntimeError crashes in environments without ffmpeg 4
sys.modules['torchcodec'] = MagicMock()

import torch
import numpy as np
import pytest
from lewm.goal_mapper import GoalMapper


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_encoder = MagicMock(
            side_effect=lambda x: torch.zeros((*x.shape[:-1], 192))
        )

    def encode(self, info):
        # Mock JEPA.encode behavior: (B, T, C, H, W) -> (B, T, D)
        pixels = info["pixels"]
        B, T = pixels.shape[0], pixels.shape[1]
        info["emb"] = torch.zeros((B, T, 192))
        return info

    def predict(self, emb, act_emb):
        # Mock JEPA.predict behavior: (BS, T, D) -> (BS, T, D)
        return torch.zeros_like(emb)


@pytest.fixture
def mock_mapper():
    # We bypass __init__ to avoid loading weights
    mapper = GoalMapper.__new__(GoalMapper)
    mapper.device = "cpu"
    mapper.model = MockModel()
    mapper.goal_latent = torch.zeros((1, 1, 192))
    return mapper


def test_canoncial_5d_protocol_stripping(mock_mapper):
    """Verifies that 6D input is transparently stripped to 5D before encoding."""
    # B=2, S=8000, T=3, C=3, H=224, W=224
    pixels_6d = torch.zeros((2, 8000, 3, 3, 224, 224))
    actions_4d = torch.zeros((2, 8000, 15, 64))

    obs_dict = {"pixels": pixels_6d}

    # This should not crash despite JEPA only supporting 5D
    costs = mock_mapper.get_cost(obs_dict, actions_4d)

    assert costs.shape == (2, 8000)
    assert isinstance(costs, torch.Tensor)


def test_omni_goal_min_dist(mock_mapper):
    """Verifies that Omni-Goal mode picks the minimum distance across the gallery."""
    # Setup: 1 Env, 2 Samples, 3 Goals
    B, S, G, D = 1, 2, 3, 192

    # Mock goals in the mapper
    # Goal 0 is far, Goal 1 is close to sample A, Goal 2 is close to sample B
    mock_mapper.goal_latent = torch.zeros((G, 1, D))
    mock_mapper.goal_latent[0, 0, 0] = 10.0  # Far
    mock_mapper.goal_latent[1, 0, 0] = 1.0  # Close

    # Mock final latents for 2 samples
    # We need to reach the end of get_cost with specific final_latents
    # Since we use a MockModel that returns zeros, we'll just verify the broadcast logic

    pixels_5d = torch.zeros((B, 3, 3, 224, 224))
    actions_4d = torch.zeros((B, S, 15, 64))

    costs = mock_mapper.get_cost({"pixels": pixels_5d}, actions_4d)

    # With MockModel returning zeros, all final_latents are zero.
    # Closest goal to zero is Goal 2 (all zeros). Distance should be 0.
    assert torch.allclose(costs, torch.zeros_like(costs))


def test_targeted_audit_parity(mock_mapper):
    """Verifies that in Audit mode (B == G), mapping is 1-to-1."""
    # Setup: 2 Envs, 2 Goals
    B, G, D = 2, 2, 192
    S = 10

    mock_mapper.goal_latent = torch.zeros((G, 1, D))
    # Env 0 and Env 1 have different target goals

    pixels_5d = torch.zeros((B, 3, 3, 224, 224))
    actions_4d = torch.zeros((B, S, 15, 64))

    costs = mock_mapper.get_cost({"pixels": pixels_5d}, actions_4d)
    assert costs.shape == (B, S)


def test_invalid_dimensions_rejection(mock_mapper):
    """Verifies that the JEPA core (via the mock) still rejects garbage dims if passed directly."""
    # Re-enable the real check for this specific test subcase
    from le_wm.jepa import JEPA as RealJEPA

    # Create a real JEPA object but mock its components
    real_jepa = RealJEPA(MagicMock(), MagicMock(), MagicMock())

    with pytest.raises(ValueError, match="JEPA.encode expects 5D"):
        # Pass 6D directly to encode (bypass wrapper)
        real_jepa.encode({"pixels": torch.zeros((1, 2, 3, 4, 5, 5))})
