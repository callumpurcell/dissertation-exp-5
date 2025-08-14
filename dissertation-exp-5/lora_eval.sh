MODELS=(
    outputs/llama32_8B_baseline_12000_gpt_2e-4_20250802_133054
    outputs/llama31_8B_reasoning_trace_longest_quartile_10k_gpt_2e-4_20250724_191316
    outputs/llama31_8B_reasoning_trace_long_quartile_10k_gpt_2e-4_20250724_191316
    outputs/llama31_8B_reasoning_trace_shortest_quartile_10k_gpt_2e-4_20250724_191316
    outputs/llama31_8B_reasoning_trace_short_quartile_10k_gpt_2e-4_20250724_191316
    outputs/llama31_8B_context_length_long_quartile_10k_gpt_2e-4_20250724_191316
    outputs/llama31_8B_context_length_longest_quartile_10k_gpt_2e-4_20250724_191316
    outputs/llama31_8B_context_length_shortest_quartile_10k_gpt_2e-4_20250724_191316
    outputs/llama31_8B_context_length_short_quartile_10k_gpt_2e-4_20250724_191316
    outputs/llama32_3B_context_length_longest_quartile_10k_gpt_2e-4_20250724_191316
    outputs/llama32_3B_context_length_long_quartile_10k_gpt_2e-4_20250724_191316
    outputs/llama32_3B_context_length_shortest_quartile_10k_gpt_2e-4_20250724_191316
    outputs/llama32_3B_context_length_short_quartile_10k_gpt_2e-4_20250724_191316
    outputs/llama32_3B_baseline_12000_gpt_2e-4_20250802_133054
    outputs/llama32_3B_reasoning_trace_longest_quartile_10k_gpt_2e-4_20250724_191316
    outputs/llama32_3B_reasoning_trace_long_quartile_10k_gpt_2e-4_20250724_191316
    outputs/llama32_3B_reasoning_trace_shortest_quartile_10k_gpt_2e-4_20250724_191316
    outputs/llama32_3B_reasoning_trace_short_quartile_10k_gpt_2e-4_20250724_191316
)
TASKS=(
    "tatqa_sft"   
    "finqa_sft"
    "synthqa"
    "synthqa_gold"
)

for MODEL_NAME in "${MODELS[@]}"; do
    for TASK_NAME in "${TASKS[@]}"; do
        accelerate launch --main_process_port 29501 -m lm_eval \
            --model "hf" \
            --model_args "pretrained=$MODEL_NAME,trust_remote_code=True" \
            --tasks "$TASK_NAME" \
            --device "cuda" \
            --batch_size "1" \
            --output_path "results" \
            --log_samples \
            --write_out \
            --apply_chat_template
    done
done
