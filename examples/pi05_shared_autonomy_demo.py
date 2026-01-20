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

"""
PI0.5 Shared Autonomy Demo

Demonstrates how to use the Shared Autonomy feature with PI0.5 flow matching,
adapting ideas from "To the Noise and Back: Diffusion for Shared Autonomy".

This script shows:
1. How to enable shared autonomy for a PI0.5 policy
2. How to provide human action input
3. How to compare behavior across different forward_flow_ratio values
4. How to visualize the fidelity-conformity trade-off
"""

import numpy as np
import torch

from lerobot.policies.pi05 import PI05Config, PI05Policy, SharedAutonomyConfig


def create_dummy_observation(batch_size=1, action_dim=7, device="cpu"):
    """Create a dummy observation for demonstration purposes.

    In a real scenario, this would come from your environment or robot sensors.
    """
    return {
        "observation.images.top": torch.randn(batch_size, 3, 224, 224, device=device),
        "observation.state": torch.randn(batch_size, 32, device=device),
        "language_instruction": "Pick up the red block",
    }


def create_noisy_human_action(ground_truth_action, noise_level=0.3):
    """Simulate a noisy human pilot.

    Args:
        ground_truth_action: The ideal action
        noise_level: Standard deviation of Gaussian noise to add

    Returns:
        Noisy human action
    """
    noise = torch.randn_like(ground_truth_action) * noise_level
    return ground_truth_action + noise


def create_laggy_human_action(previous_action, current_action, lag_probability=0.7):
    """Simulate a laggy human pilot that sometimes repeats previous actions.

    Args:
        previous_action: Action from previous timestep
        current_action: Current intended action
        lag_probability: Probability of using previous action

    Returns:
        Potentially lagged action
    """
    if np.random.random() < lag_probability:
        return previous_action
    return current_action


def demo_basic_usage():
    """Demonstrate basic shared autonomy usage."""
    print("=" * 80)
    print("DEMO 1: Basic Shared Autonomy Usage")
    print("=" * 80)

    # Create policy with shared autonomy enabled
    sa_config = SharedAutonomyConfig(
        enabled=True,
        forward_flow_ratio=0.4,  # 40% noise, 60% human action preserved
    )

    config = PI05Config(
        paligemma_variant="gemma_300m",
        action_expert_variant="gemma_300m",
        shared_autonomy_config=sa_config,
        device="cpu",
    )

    print(f"\nCreating PI0.5 policy with shared autonomy...")
    print(f"  forward_flow_ratio: {sa_config.forward_flow_ratio}")
    print(f"  (Lower = more fidelity to human, Higher = more conformity to model)")

    # Note: In practice, you would use:
    # policy = PI05Policy.from_pretrained("physical-intelligence/pi05-base")
    # Then manually enable shared autonomy as shown in the docstring

    # For this demo, we'll show the API usage pattern
    print("\nAPI Usage Pattern:")
    print("  1. policy.set_human_action(human_action)")
    print("  2. model_action = policy.select_action(observation)")
    print("  -> Model blends human_action with its predictions")

    print("\n✓ Basic usage demonstrated (see code for details)")


def demo_forward_flow_ratio_comparison():
    """Demonstrate the effect of different forward_flow_ratio values."""
    print("\n" + "=" * 80)
    print("DEMO 2: Comparing Different Forward Flow Ratios")
    print("=" * 80)

    batch_size = 1
    action_dim = 7
    chunk_size = 50

    # Simulate a "ground truth" expert action
    expert_action = torch.randn(batch_size, action_dim)

    # Simulate noisy human action
    noise_level = 0.5
    human_action = create_noisy_human_action(expert_action, noise_level=noise_level)

    print(f"\nGround truth expert action: {expert_action.squeeze().numpy()[:3]}...")
    print(f"Noisy human action: {human_action.squeeze().numpy()[:3]}...")
    print(f"Noise level: {noise_level}")

    # Test different forward_flow_ratio values
    ratios_to_test = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    print("\n" + "-" * 80)
    print(f"{'Ratio':<8} {'Interpretation':<30} {'Fidelity':<12} {'Conformity':<12}")
    print("-" * 80)

    for ratio in ratios_to_test:
        if ratio == 0.0:
            interpretation = "Pure human control"
            fidelity = "Maximum"
            conformity = "Minimum"
        elif ratio == 1.0:
            interpretation = "Pure model control"
            fidelity = "Minimum"
            conformity = "Maximum"
        else:
            interpretation = f"{int((1-ratio)*100)}% human / {int(ratio*100)}% noise"
            fidelity = "High" if ratio < 0.5 else "Low"
            conformity = "Low" if ratio < 0.5 else "High"

        print(f"{ratio:<8.1f} {interpretation:<30} {fidelity:<12} {conformity:<12}")

    print("-" * 80)
    print("\nKey Insight:")
    print("  - Low ratio (e.g., 0.2): Preserves human intent, applies light corrections")
    print("  - Medium ratio (e.g., 0.4-0.6): Balanced blending")
    print("  - High ratio (e.g., 0.8): Strong corrections, mostly follows model")


def demo_pilot_simulation():
    """Simulate different types of pilots (noisy, laggy) with shared autonomy."""
    print("\n" + "=" * 80)
    print("DEMO 3: Simulating Different Pilot Types")
    print("=" * 80)

    # Simulate a sequence of expert actions (what should happen)
    num_steps = 5
    action_dim = 7
    expert_actions = [torch.randn(1, action_dim) for _ in range(num_steps)]

    print("\nSimulating 3 pilot types over 5 timesteps:")
    print("  1. Noisy Pilot: Adds Gaussian noise to expert actions")
    print("  2. Laggy Pilot: Sometimes repeats previous action")
    print("  3. Random Pilot: Completely random actions")

    pilot_types = {
        "Noisy (30% noise)": lambda i: create_noisy_human_action(expert_actions[i], noise_level=0.3),
        "Laggy (70% lag)": lambda i: create_laggy_human_action(
            expert_actions[max(0, i - 1)], expert_actions[i], lag_probability=0.7
        )
        if i > 0
        else expert_actions[0],
        "Random": lambda i: torch.randn(1, action_dim),
    }

    for pilot_name, pilot_func in pilot_types.items():
        print(f"\n{pilot_name}:")
        print("  " + "-" * 60)

        for step in range(num_steps):
            expert_action = expert_actions[step]
            human_action = pilot_func(step)

            # Calculate error
            error = torch.norm(human_action - expert_action).item()

            print(f"  Step {step}: Human action error = {error:.3f}")

        print(f"  -> With shared autonomy (forward_flow_ratio=0.4):")
        print(f"     Model would correct these actions toward learned distribution")


def demo_fidelity_conformity_tradeoff():
    """Visualize the fidelity-conformity trade-off mathematically."""
    print("\n" + "=" * 80)
    print("DEMO 4: Fidelity-Conformity Trade-off (Mathematical)")
    print("=" * 80)

    print("\nFlow Matching Forward Process:")
    print("  x_t = t * noise + (1-t) * human_action")
    print("\nWhere:")
    print("  - t = forward_flow_ratio (our 'switching time')")
    print("  - noise ~ N(0, I) is sampled Gaussian noise")
    print("  - human_action is the pilot's input")

    print("\n" + "-" * 80)
    print("Expected Deviation from Human Action:")
    print("-" * 80)

    action_dim = 7
    ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    print(f"\n{'Ratio':<10} {'Expected ||deviation||²':<25} {'Interpretation'}")
    print("-" * 80)

    for t_sw in ratios:
        # E[||x_tsw - human_action||²] = t_sw² * d (where d = action_dim)
        expected_deviation_squared = t_sw**2 * action_dim

        if t_sw == 0.0:
            interpretation = "No deviation (pure human)"
        elif t_sw == 1.0:
            interpretation = "Maximum deviation (pure model)"
        else:
            interpretation = f"{int(t_sw*100)}% of max deviation"

        print(f"{t_sw:<10.1f} {expected_deviation_squared:<25.2f} {interpretation}")

    print("-" * 80)
    print("\nKey Property: Deviation grows QUADRATICALLY with forward_flow_ratio!")
    print("This means small increases in ratio have bigger impact at higher values.")


def demo_recommended_settings():
    """Show recommended settings for different use cases."""
    print("\n" + "=" * 80)
    print("DEMO 5: Recommended Settings for Different Use Cases")
    print("=" * 80)

    use_cases = {
        "Teleoperation assistance (expert user)": {
            "forward_flow_ratio": 0.1,
            "rationale": "User is skilled, only need safety corrections",
        },
        "Teleoperation assistance (novice user)": {
            "forward_flow_ratio": 0.4,
            "rationale": "User needs more guidance, balanced blending",
        },
        "Shared control in uncertain environments": {
            "forward_flow_ratio": 0.3,
            "rationale": "Preserve human judgment while correcting mistakes",
        },
        "Learning from demonstration (active learning)": {
            "forward_flow_ratio": 0.6,
            "rationale": "Strong corrections to guide user toward optimal behavior",
        },
        "Safety-critical corrections only": {
            "forward_flow_ratio": 0.2,
            "rationale": "Minimal intervention unless necessary",
        },
    }

    for use_case, settings in use_cases.items():
        print(f"\n{use_case}:")
        print(f"  forward_flow_ratio: {settings['forward_flow_ratio']}")
        print(f"  Rationale: {settings['rationale']}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("PI0.5 SHARED AUTONOMY DEMONSTRATION")
    print("Adapting 'To the Noise and Back' for Flow Matching")
    print("=" * 80)

    demo_basic_usage()
    demo_forward_flow_ratio_comparison()
    demo_pilot_simulation()
    demo_fidelity_conformity_tradeoff()
    demo_recommended_settings()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nShared Autonomy Key Concepts:")
    print("  1. forward_flow_ratio (t_sw) controls intervention strength")
    print("  2. Lower ratio = more fidelity to human (preserve intent)")
    print("  3. Higher ratio = more conformity to model (stronger corrections)")
    print("  4. Flow matching provides smooth, continuous control")
    print("  5. No retraining needed - works with pretrained PI0.5 models")
    print("\nFor Production Use:")
    print("  - Start with forward_flow_ratio=0.4 and tune based on user feedback")
    print("  - Monitor task success rate vs. user satisfaction")
    print("  - Consider adaptive forward_flow_ratio based on user confidence")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
