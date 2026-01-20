#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for PI0.5 Shared Autonomy implementation."""

import pytest
import torch

from lerobot.policies.pi05 import PI05Config, SharedAutonomyConfig, SharedAutonomyProcessor


class TestSharedAutonomyConfig:
    """Test SharedAutonomyConfig validation and initialization."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = SharedAutonomyConfig()

        assert config.enabled is False
        assert config.forward_flow_ratio == 0.4
        assert config.human_action_buffer_size == 1
        assert config.apply_to_first_action_only is True
        assert config.debug is False

    def test_config_validation_forward_flow_ratio(self):
        """Test forward_flow_ratio validation."""
        # Valid values
        SharedAutonomyConfig(forward_flow_ratio=0.0)
        SharedAutonomyConfig(forward_flow_ratio=0.5)
        SharedAutonomyConfig(forward_flow_ratio=1.0)

        # Invalid values
        with pytest.raises(ValueError, match="forward_flow_ratio must be in"):
            SharedAutonomyConfig(forward_flow_ratio=-0.1)

        with pytest.raises(ValueError, match="forward_flow_ratio must be in"):
            SharedAutonomyConfig(forward_flow_ratio=1.1)

    def test_config_validation_buffer_size(self):
        """Test human_action_buffer_size validation."""
        # Valid values
        SharedAutonomyConfig(human_action_buffer_size=1)
        SharedAutonomyConfig(human_action_buffer_size=10)

        # Invalid values
        with pytest.raises(ValueError, match="human_action_buffer_size must be positive"):
            SharedAutonomyConfig(human_action_buffer_size=0)

        with pytest.raises(ValueError, match="human_action_buffer_size must be positive"):
            SharedAutonomyConfig(human_action_buffer_size=-1)


class TestSharedAutonomyProcessor:
    """Test SharedAutonomyProcessor functionality."""

    def test_processor_initialization(self):
        """Test processor initializes correctly."""
        config = SharedAutonomyConfig(enabled=True, forward_flow_ratio=0.3)
        processor = SharedAutonomyProcessor(config)

        assert processor.sa_config == config
        assert processor.has_human_action() is False

    def test_human_action_buffer(self):
        """Test human action buffering."""
        config = SharedAutonomyConfig(human_action_buffer_size=2)
        processor = SharedAutonomyProcessor(config)

        # Initially empty
        assert processor.has_human_action() is False
        assert processor.get_human_action() is None

        # Add actions
        action1 = torch.randn(1, 7)
        action2 = torch.randn(1, 7)

        processor.set_human_action(action1)
        assert processor.has_human_action() is True

        processor.set_human_action(action2)
        assert processor.has_human_action() is True

        # Retrieve in FIFO order
        retrieved1 = processor.get_human_action()
        assert torch.equal(retrieved1, action1)

        retrieved2 = processor.get_human_action()
        assert torch.equal(retrieved2, action2)

        # Now empty again
        assert processor.has_human_action() is False

    def test_compute_initial_state_3d(self):
        """Test partial forward flow computation with 3D tensors."""
        config = SharedAutonomyConfig(forward_flow_ratio=0.4)
        processor = SharedAutonomyProcessor(config)

        batch_size, chunk_size, action_dim = 2, 50, 7
        noise = torch.randn(batch_size, chunk_size, action_dim)
        human_action = torch.randn(batch_size, chunk_size, action_dim)

        x_tsw = processor.compute_initial_state(noise, human_action)

        # Verify shape
        assert x_tsw.shape == (batch_size, chunk_size, action_dim)

        # Verify forward flow formula: x_t = t * noise + (1-t) * action
        expected = 0.4 * noise + 0.6 * human_action
        assert torch.allclose(x_tsw, expected)

    def test_compute_initial_state_2d_broadcast(self):
        """Test partial forward flow with 2D human action (broadcasted to chunk)."""
        config = SharedAutonomyConfig(forward_flow_ratio=0.5)
        processor = SharedAutonomyProcessor(config)

        batch_size, chunk_size, action_dim = 2, 50, 7
        noise = torch.randn(batch_size, chunk_size, action_dim)
        human_action = torch.randn(batch_size, action_dim)  # 2D

        x_tsw = processor.compute_initial_state(noise, human_action)

        # Verify shape
        assert x_tsw.shape == (batch_size, chunk_size, action_dim)

        # Verify broadcasting and formula
        human_action_expanded = human_action.unsqueeze(1).expand_as(noise)
        expected = 0.5 * noise + 0.5 * human_action_expanded
        assert torch.allclose(x_tsw, expected)

    def test_compute_initial_state_shape_mismatch(self):
        """Test that shape mismatch raises error."""
        config = SharedAutonomyConfig()
        processor = SharedAutonomyProcessor(config)

        noise = torch.randn(2, 50, 7)
        human_action = torch.randn(2, 10)  # Wrong action_dim

        with pytest.raises(ValueError, match="Shape mismatch"):
            processor.compute_initial_state(noise, human_action)

    def test_modify_denoising_params(self):
        """Test denoising parameter modification."""
        config = SharedAutonomyConfig(forward_flow_ratio=0.4)
        processor = SharedAutonomyProcessor(config)

        num_steps = 10
        t_start, dt, returned_steps = processor.modify_denoising_params(num_steps)

        # Verify values
        assert t_start == 0.4
        assert dt == -0.04  # -0.4 / 10
        assert returned_steps == 10

        # Test different forward_flow_ratio
        config2 = SharedAutonomyConfig(forward_flow_ratio=0.8)
        processor2 = SharedAutonomyProcessor(config2)

        t_start2, dt2, _ = processor2.modify_denoising_params(10)
        assert t_start2 == 0.8
        assert dt2 == -0.08

    def test_fidelity_extremes(self):
        """Test fidelity at extreme forward_flow_ratio values."""
        batch_size, chunk_size, action_dim = 1, 50, 7
        noise = torch.randn(batch_size, chunk_size, action_dim)
        human_action = torch.randn(batch_size, chunk_size, action_dim)

        # t_sw = 0.0: Maximum fidelity (return human action)
        config_zero = SharedAutonomyConfig(forward_flow_ratio=0.0)
        processor_zero = SharedAutonomyProcessor(config_zero)
        x_tsw_zero = processor_zero.compute_initial_state(noise, human_action)
        assert torch.allclose(x_tsw_zero, human_action)

        # t_sw = 1.0: Maximum conformity (pure noise)
        config_one = SharedAutonomyConfig(forward_flow_ratio=1.0)
        processor_one = SharedAutonomyProcessor(config_one)
        x_tsw_one = processor_one.compute_initial_state(noise, human_action)
        assert torch.allclose(x_tsw_one, noise)

    def test_interpolation_property(self):
        """Test that x_tsw lies between noise and human_action."""
        batch_size, chunk_size, action_dim = 1, 50, 7

        # Use deterministic values for easier testing
        noise = torch.ones(batch_size, chunk_size, action_dim)
        human_action = torch.zeros(batch_size, chunk_size, action_dim)

        config = SharedAutonomyConfig(forward_flow_ratio=0.3)
        processor = SharedAutonomyProcessor(config)

        x_tsw = processor.compute_initial_state(noise, human_action)

        # x_tsw should be 0.3 * 1 + 0.7 * 0 = 0.3
        expected = torch.full((batch_size, chunk_size, action_dim), 0.3)
        assert torch.allclose(x_tsw, expected)


class TestPI05ConfigIntegration:
    """Test PI05Config integration with SharedAutonomyConfig."""

    def test_pi05_config_with_shared_autonomy(self):
        """Test PI05Config can include shared_autonomy_config."""
        sa_config = SharedAutonomyConfig(
            enabled=True,
            forward_flow_ratio=0.5,
        )

        pi05_config = PI05Config(
            shared_autonomy_config=sa_config,
        )

        assert pi05_config.shared_autonomy_config is not None
        assert pi05_config.shared_autonomy_config.enabled is True
        assert pi05_config.shared_autonomy_config.forward_flow_ratio == 0.5

    def test_pi05_config_without_shared_autonomy(self):
        """Test PI05Config works without shared_autonomy_config."""
        pi05_config = PI05Config()

        assert pi05_config.shared_autonomy_config is None


@pytest.mark.parametrize("forward_flow_ratio", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
def test_forward_flow_ratio_sweep(forward_flow_ratio):
    """Test different forward_flow_ratio values produce expected blending."""
    config = SharedAutonomyConfig(forward_flow_ratio=forward_flow_ratio)
    processor = SharedAutonomyProcessor(config)

    batch_size, chunk_size, action_dim = 1, 50, 7
    noise = torch.randn(batch_size, chunk_size, action_dim)
    human_action = torch.randn(batch_size, chunk_size, action_dim)

    x_tsw = processor.compute_initial_state(noise, human_action)

    # Verify formula
    expected = forward_flow_ratio * noise + (1 - forward_flow_ratio) * human_action
    assert torch.allclose(x_tsw, expected, atol=1e-6)

    # Verify denoising params
    t_start, dt, _ = processor.modify_denoising_params(10)
    assert t_start == forward_flow_ratio
    assert abs(dt - (-forward_flow_ratio / 10)) < 1e-6
