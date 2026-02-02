#!/bin/bash

# Script to run lerobot-eval on all policy checkpoints
# Handles diffusion_approach_lever_* and pi05_training_approach_lever_* folders

OUTPUTS_DIR="/home/jennyw2/code/lerobot/outputs"
TIMESTAMP=$(date +"%Y-%m-%d-%H%M%S")
EVAL_OUTPUT_DIR="$OUTPUTS_DIR/eval_output/$TIMESTAMP"
EXP_PATTERNS=("$OUTPUTS_DIR"/diffusion_approach_lever_* "$OUTPUTS_DIR"/pi05_training_approach_lever_*)

# Collect all experiment folders that will be processed
echo "========================================"
echo "EVALUATION PLAN"
echo "========================================"
echo ""
echo "Experiment folders to process:"
echo ""

eval_count=0
for exp_dir in "${EXP_PATTERNS[@]}"; do
    [ -d "$exp_dir" ] || continue
    exp_name=$(basename "$exp_dir")
    checkpoints_dir="$exp_dir/checkpoints"
    [ -d "$checkpoints_dir" ] || continue

    # Count checkpoints (excluding last numerical which equals 'last')
    last_numerical=$(ls -1 "$checkpoints_dir" | grep -E '^[0-9]+$' | sort -n | tail -1)
    checkpoint_count=0
    for checkpoint_dir in "$checkpoints_dir"/*; do
        [ -d "$checkpoint_dir" ] || continue
        checkpoint_name=$(basename "$checkpoint_dir")
        [ "$checkpoint_name" == "$last_numerical" ] && continue
        [ -d "$checkpoint_dir/pretrained_model" ] || continue
        ((checkpoint_count++))
    done

    echo "  - $exp_name ($checkpoint_count checkpoints)"
    ((eval_count += checkpoint_count))
done

echo ""
echo "Total evaluations to run: $eval_count"
echo "========================================"
echo ""

# Function to determine camera names based on experiment suffix
get_camera_names() {
    local exp_name="$1"

    # Check for basewrist variants (must check before base/wrist alone)
    if [[ "$exp_name" == *"_basewrist"* ]] || [[ "$exp_name" == *"_basewristrgb"* ]]; then
        echo '["base_rgb", "wrist_rgb"]'
    # Check for base only
    elif [[ "$exp_name" == *"_base"* ]] || [[ "$exp_name" == *"_basergb"* ]]; then
        echo '["base_rgb"]'
    # Check for wrist only
    elif [[ "$exp_name" == *"_wrist"* ]] || [[ "$exp_name" == *"_wristrgb"* ]]; then
        echo '["wrist_rgb"]'
    else
        # Default to base_rgb if unclear
        echo '["base_rgb"]'
    fi
}

# Find the last numerical checkpoint (to skip it since 'last' symlink points to it)
get_last_numerical_checkpoint() {
    local checkpoints_dir="$1"
    # Get all numerical checkpoint directories, sort them, and get the last one
    ls -1 "$checkpoints_dir" | grep -E '^[0-9]+$' | sort -n | tail -1
}

# Process each matching experiment folder
for exp_dir in "${EXP_PATTERNS[@]}"; do
    # Skip if not a directory
    [ -d "$exp_dir" ] || continue

    exp_name=$(basename "$exp_dir")
    checkpoints_dir="$exp_dir/checkpoints"

    # Skip if no checkpoints directory
    if [ ! -d "$checkpoints_dir" ]; then
        echo "Skipping $exp_name: no checkpoints directory"
        continue
    fi

    # Get camera names for this experiment
    camera_names=$(get_camera_names "$exp_name")

    # Find the last numerical checkpoint to skip
    last_numerical=$(get_last_numerical_checkpoint "$checkpoints_dir")

    echo "========================================"
    echo "Processing experiment: $exp_name"
    echo "Camera names: $camera_names"
    echo "Last numerical checkpoint (will skip): $last_numerical"
    echo "========================================"

    # Process each checkpoint
    index=0
    for checkpoint_dir in "$checkpoints_dir"/*; do
        [ -d "$checkpoint_dir" ] || continue

        # Skip first checkpoint for debugging
        if [ $index -eq 0 ]; then
            ((index++))
            continue
        fi
        ((index++))

        checkpoint_name=$(basename "$checkpoint_dir")

        # Skip the last numerical checkpoint (it's the same as 'last')
        if [ "$checkpoint_name" == "$last_numerical" ]; then
            echo "Skipping $checkpoint_name (same as 'last')"
            continue
        fi

        policy_path="$checkpoint_dir/pretrained_model"

        # Check if pretrained_model exists
        if [ ! -d "$policy_path" ]; then
            echo "Skipping $checkpoint_name: no pretrained_model directory"
            continue
        fi

        # Create unique output directory for this evaluation
        eval_subdir="$EVAL_OUTPUT_DIR/${exp_name}_${checkpoint_name}"
        mkdir -p "$eval_subdir"

        # Log file for this evaluation
        log_file="$eval_subdir/eval_log.txt"

        echo "----------------------------------------"
        echo "Running eval for: $exp_name / $checkpoint_name"
        echo "Policy path: $policy_path"
        echo "Output dir: $eval_subdir"
        echo "Log file: $log_file"
        echo "----------------------------------------"

        # Run eval and capture output to both terminal and log file
        # Filter out progress bar lines from the log file
        {
            echo "========================================"
            echo "Evaluation Log"
            echo "Experiment: $exp_name"
            echo "Checkpoint: $checkpoint_name"
            echo "Policy path: $policy_path"
            echo "Camera names: $camera_names"
            echo "Start time: $(date)"
            echo "========================================"
            echo ""

            eval_cmd="lerobot-eval \\
                --env.type=splatsim \\
                --env.task=upright_small_engine_new \\
                --env.camera_names='$camera_names' \\
                --env.fps=30 \\
                --policy.path=$policy_path \\
                --eval.n_episodes=5 \\
                --output_dir=$eval_subdir \\
                --eval.batch_size=1 \\
                --eval.use_async_envs=false"

            echo "Command:"
            echo "$eval_cmd"
            echo ""

            eval "$eval_cmd"

            echo ""
            echo "========================================"
            echo "End time: $(date)"
            echo "========================================"
        } 2>&1 | tee >(grep -v "Running rollout with at most 200 steps:" > "$log_file")

        echo "Completed: $exp_name / $checkpoint_name"
        echo ""
    done
done

echo "========================================"
echo "All evaluations complete!"
echo "========================================"
