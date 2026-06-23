# cd ~/.cache/huggingface/lerobot/JennyWWW
# for d in approach_lever_11_biasend_5path_grip0_dag*; do
#     mv "$d" "${d%_dag*}_a${d#${d%_dag*}}"
# done
cd ~/code/lerobot/outputs/dataset_stats
for d in approach_lever_11_biasend_5path_grip0_dag*; do
    mv "$d" "${d%_dag*}_a${d#${d%_dag*}}"
done


# Now run the rel orchestration fresh (no force_restart needed; resume detection
# will see no existing dag artifacts under the rel run's lineage).
bash my_scripts/dagger_orchestrate.sh \
  --base_short=approach_lever_11_biasend_5path_grip0 \
  --num_rounds=10 \
  --action_format=rel \
  --intermediate_mode=finetune --final_mode=scratch \
  --intervention_n_episodes=5 --intervention_oversample=3 \
  --finetune_steps=1000 --finetune_eval_freq=500 --finetune_save_freq=500 \
  --finetune_decay_lr=5e-6 \
  --env_external_port=6001 \
  --skip_alias_step
