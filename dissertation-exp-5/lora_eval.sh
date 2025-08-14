#export CUDA_VISIBLE_DEVICES=4,5,6,7
MODELS=(
    # ../outputs/llama32_1B_question_diversity_highest_quartile_1k_gpt_2e-4_20250720_180720
    # ../outputs/llama32_1B_question_diversity_lower_middle_quartile_1k_gpt_2e-4_20250720_180720
    # ../outputs/llama32_1B_question_diversity_lowest_quartile_1k_gpt_2e-4_20250720_180720
    # ../outputs/llama32_1B_question_diversity_upper_middle_quartile_1k_gpt_2e-4_20250720_180720
    # ../outputs/llama32_1B_question_diversity_highest_quartile_5k_gpt_2e-4_20250720_180720
    # ../outputs/llama32_1B_question_diversity_lower_middle_quartile_5k_gpt_2e-4_20250720_180720
    # ../outputs/llama32_1B_question_diversity_lowest_quartile_5k_gpt_2e-4_20250720_180720
    # ../outputs/llama32_1B_question_diversity_upper_middle_quartile_5k_gpt_2e-4_20250720_180720
    # ../outputs/llama32_1B_lora-8_2e-3_20250720_130715
    # ../outputs/llama32_1B_lora-8_2e-4_20250720_130715
    # ../outputs/llama32_1B_lora-8_2e-5_20250720_130715
    # ../outputs/llama32_1B_lora-16_2e-3_20250720_134651
    # ../outputs/llama32_1B_lora-16_2e-4_20250720_134651
    # ../outputs/llama32_1B_lora-16_2e-5_20250720_134651
    # ../outputs/llama32_1B_lora-32_2e-3_20250720_150739
    # ../outputs/llama32_1B_lora-32_2e-4_20250720_150739
    # ../outputs/llama32_1B_lora-32_2e-5_20250720_150739
    # ../outputs/llama32_1B_low_context_diversity_1k_gpt_2e-4_20250722_080908
    # # ../outputs/llama32_1B_low_context_diversity_5k_gpt_2e-4_20250722_080908
    # ../outputs/llama32_1B_high_context_diversity_1k_gpt_2e-4_20250722_080908
    # # ../outputs/llama32_1B_high_context_diversity_5k_gpt_2e-4_20250722_080908
    # ../outputs/llama32_1B_context_length_long_quartile_1k_gpt_2e-4_20250722_112829
    # ../outputs/llama32_1B_context_length_short_quartile_1k_gpt_2e-4_20250722_112829
    # ../outputs/llama32_1B_context_length_long_quartile_5k_gpt_2e-4_20250722_112829
    # ../outputs/llama32_1B_context_length_short_quartile_5k_gpt_2e-4_20250722_112829
    # ../outputs/llama32_1B_context_length_longest_quartile_1k_gpt_2e-4_20250722_112829
    # ../outputs/llama32_1B_context_length_longest_quartile_5k_gpt_2e-4_20250722_112829
    # ../outputs/llama32_1B_context_length_shortest_quartile_1k_gpt_2e-4_20250722_112829
    # ../outputs/llama32_1B_context_length_shortest_quartile_5k_gpt_2e-4_20250722_112829
    # ../outputs/llama32_1B_reasoning_trace_longest_quartile_1k_gpt_2e-4_20250722_112829
    # ../outputs/llama32_1B_reasoning_trace_longest_quartile_5k_gpt_2e-4_20250722_112829
    # ../outputs/llama32_1B_reasoning_trace_shortest_quartile_1k_gpt_2e-4_20250722_112829
    # ../outputs/llama32_1B_reasoning_trace_shortest_quartile_5k_gpt_2e-4_20250722_112829
    # ../outputs/llama32_1B_reasoning_trace_long_quartile_1k_gpt_2e-4_20250722_112829
    # ../outputs/llama32_1B_reasoning_trace_long_quartile_5k_gpt_2e-4_20250722_112829
    # ../outputs/llama32_1B_reasoning_trace_short_quartile_1k_gpt_2e-4_20250722_112829
    # ../outputs/llama32_1B_reasoning_trace_short_quartile_5k_gpt_2e-4_20250722_112829
    # ../outputs_surtr/outputs/llama32_1B_context_length_long_quartile_10k_gpt_2e-4_20250724_191316
    # ../outputs_surtr/outputs/llama32_1B_context_length_short_quartile_10k_gpt_2e-4_20250724_191316
    # ../outputs_surtr/outputs/llama32_1B_context_length_longest_quartile_10k_gpt_2e-4_20250724_191316
    # ../outputs_surtr/outputs/llama32_1B_context_length_shortest_quartile_10k_gpt_2e-4_20250724_191316
    # ../outputs_surtr/outputs/llama32_1B_reasoning_trace_longest_quartile_10k_gpt_2e-4_20250724_191316
    # ../outputs_surtr/outputs/llama32_1B_reasoning_trace_shortest_quartile_10k_gpt_2e-4_20250724_191316
    # ../outputs_surtr/outputs/llama32_1B_reasoning_trace_long_quartile_10k_gpt_2e-4_20250724_191316
    # ../outputs_surtr/outputs/llama32_1B_reasoning_trace_short_quartile_10k_gpt_2e-4_20250724_191316
    # ../outputs_surtr/outputs/llama32_1B_question_diversity_highest_quartile_10k_gpt_2e-4_20250724_191316
    # ../outputs_surtr/outputs/llama32_1B_question_diversity_lowest_quartile_10k_gpt_2e-4_20250724_191316
    # meta-llama/Llama-3.1-8B-Instruct
    # ../outputs_surtr/outputs/llama32_1B_question_diversity_upper_middle_quartile_10k_gpt_2e-4_20250725_100943
    # ../outputs_surtr/outputs/llama32_1B_question_diversity_lower_middle_quartile_10k_gpt_2e-4_20250725_100943
    # ../outputs_surtr/outputs/llama32_1B_high_context_diversity_10k_gpt_2e-4_20250725_100943
    # ../outputs_surtr/outputs/llama32_1B_low_context_diversity_10k_gpt_2e-4_20250725_100943
    # ../outputs_surtr/llama32_1B_baseline_1200_gpt_2e-4_20250725_104436
    # ../outputs_surtr/llama32_1B_baseline_6000_gpt_2e-4_20250725_104436
    # ../outputs_surtr/llama32_1B_baseline_12000_gpt_2e-4_20250725_104436
    # ../outputs_surtr/llama32_3B_baseline_1200_gpt_2e-4_20250725_170537
    # ../outputs_surtr/llama32_3B_baseline_6000_gpt_2e-4_20250725_170537
    # ../outputs_surtr/llama32_3B_baseline_12000_gpt_2e-4_20250725_170537
    # ../outputs_surtr/llama32_3B_baseline_1200_mistral_large_2e-4_20250726_141433
    # ../outputs_surtr/llama32_3B_baseline_6000_mistral_large_2e-4_20250726_141433
    # ../outputs_surtr/llama32_3B_baseline_12000_mistral_large_2e-4_20250726_141433
    # ../outputs_surtr/llama32_1B_num_pc_control_1200_gpt_2e-4_20250727_111014
    # ../outputs_surtr/llama32_1B_num_pc_control_12000_gpt_2e-4_20250727_111014
    # ../outputs_surtr/llama32_1B_num_pc_control_6000_gpt_2e-4_20250727_111014
    # ../outputs_surtr/llama32_1B_table_size_control_1200_gpt_2e-4_20250727_111014
    # ../outputs_surtr/llama32_1B_table_size_control_12000_gpt_2e-4_20250727_111014
    # ../outputs_surtr/llama32_1B_table_size_control_6000_gpt_2e-4_20250727_111014
    # ../outputs/llama32_1B_num_pc_control_12000_gpt_2e-4_20250802_133054
    # ../outputs/llama32_1B_num_pc_control_6000_gpt_2e-4_20250802_133054
    # ../outputs/llama32_1B_table_size_control_12000_gpt_2e-4_20250802_133054
    # ../outputs/llama32_1B_table_size_control_6000_gpt_2e-4_20250802_133054
    # ../outputs/llama32_1B_baseline_12000_gpt_2e-4_20250802_133054
    # ../outputs/llama32_1B_baseline_6000_gpt_2e-4_20250802_133054
    # ../outputs/llama32_1B_baseline_6000_mistral_large_2e-4_20250802_133054
    # ../outputs/llama32_1B_baseline_12000_mistral_large_2e-4_20250802_133054
    # ../outputs/llama32_1B_low_context_diversity_5k_gpt_2e-4_20250802_133054
    # ../outputs/llama32_1B_low_context_diversity_10k_gpt_2e-4_20250802_133054
    # ../outputs/llama32_1B_high_context_diversity_10k_gpt_2e-4_20250802_133054
    # ../outputs/llama32_1B_high_context_diversity_5k_gpt_2e-4_20250802_133054
    # ../outputs/llama32_1B_mistral_3B_12000_2e-4_20250802_134220
    # ../outputs/llama32_1B_mistral_3B_6000_2e-4_20250802_134220
    # ../outputs/llama32_1B_mistral_8B_12000_2e-4_20250802_134220
    # ../outputs/llama32_1B_mistral_8B_6000_2e-4_20250802_134220
    # ../outputs/llama32_1B_mistral_medium_12000_2e-4_20250802_134220
    # ../outputs/llama32_1B_mistral_medium_6000_2e-4_20250802_134220
    # ../outputs/llama32_1B_mistral_small_6000_2e-4_20250802_134220
    # ../outputs/llama32_1B_mistral_small_12000_2e-4_20250802_134220
    # ../outputs/llama32_1B_50-qs_12000_2e-4_20250803_135052
    # ../outputs/llama32_1B_50-qs_6000_2e-4_20250803_135052
    # ../outputs/llama32_1B_20-qs_12000_2e-4_20250803_135052
    # ../outputs/llama32_1B_20-qs_6000_2e-4_20250803_135052
    # ../outputs/llama32_1B_10-qs_12000_2e-4_20250803_135052
    # ../outputs/llama32_1B_10-qs_6000_2e-4_20250803_135052
    # ../outputs/llama32_1B_agreement-2-models_12000_2e-4_20250803_175519
    # ../outputs/llama32_1B_agreement-2-models_6000_2e-4_20250803_175519
    # ../outputs/llama32_1B_question_diversity_3gram__highest_quartile_10k_gpt_2e-4_20250803_181006
    # ../outputs/llama32_1B_question_diversity_3gram__highest_quartile_5k_gpt_2e-4_20250803_181006
    # ../outputs/llama32_1B_question_diversity_3gram__lowest_quartile_10k_gpt_2e-4_20250803_181006
    # ../outputs/llama32_1B_question_diversity_3gram__lowest_quartile_5k_gpt_2e-4_20250803_181006
    # ../outputs/llama32_1B_question_diversity_3gram__lower_middle_quartile_10k_gpt_2e-4_20250803_181006
    # ../outputs/llama32_1B_question_diversity_3gram__lower_middle_quartile_5k_gpt_2e-4_20250803_181006
    # ../outputs/llama32_1B_question_diversity_3gram__upper_middle_quartile_10k_gpt_2e-4_20250803_181006
    # ../outputs/llama32_1B_question_diversity_3gram__upper_middle_quartile_5k_gpt_2e-4_20250803_181006
    # ../outputs/llama32_1B_10-qs-clean_12000_2e-4_20250805_071134
    # ../outputs/llama32_1B_10-qs-clean_6000_2e-4_20250805_071134
    # ../outputs/llama32_1B_20-qs-clean_12000_2e-4_20250805_071134
    # ../outputs/llama32_1B_20-qs-clean_6000_2e-4_20250805_071134
    # ../outputs/llama32_1B_50-qs-clean_12000_2e-4_20250805_071134
    # ../outputs/llama32_1B_50-qs-clean_6000_2e-4_20250805_071134
    # ../outputs/llama32_1B_num_60_control_12000_2e-4_20250807_185251
    # ../outputs/llama32_1B_num_60_control_6000_2e-4_20250807_185251
    # ../outputs/llama32_1B_10-qs-clean_1200_2e-4_20250805_132838
    # ../outputs/llama32_1B_20-qs-clean_1200_2e-4_20250805_132838
    # ../outputs/llama32_1B_50-qs-clean_1200_2e-4_20250805_132838 
    ../outputs/llama32_1B_10-qs_1200_2e-4_20250813_210904
    ../outputs/llama32_1B_20-qs_1200_2e-4_20250813_210904
    ../outputs/llama32_1B_50-qs_1200_2e-4_20250813_210904
    ../outputs/llama32_1B_question_diversity_3gram__upper_middle_quartile_1k_gpt_2e-4_20250813_210904
    ../outputs/llama32_1B_question_diversity_3gram__lower_middle_quartile_1k_gpt_2e-4_20250813_210904
    ../outputs/llama32_1B_question_diversity_3gram__lowest_quartile_1k_gpt_2e-4_20250813_210904
    ../outputs/llama32_1B_question_diversity_3gram__highest_quartile_1k_gpt_2e-4_20250813_210904
    ../outputs/llama32_1B_num_60_control_1200_2e-4_20250813_211405
    ../outputs/llama32_1B_agreement-2-models_1200_2e-4_20250813_211405
    ../outputs/llama32_1B_mistral_3B_1200_2e-4_20250813_211405
    ../outputs/llama32_1B_mistral_8B_1200_2e-4_20250813_211405
    ../outputs/llama32_1B_mistral_medium_1200_2e-4_20250813_211405
    ../outputs/llama32_1B_mistral_small_1200_2e-4_20250813_211405
)
TASKS=(
    "tatqa_sft"   
    "finqa_sft"
    # "tathqa_sft"
    # "multihiertt_easy_sft"
    # "convfinqa_sft"
    "synthqa"
    "synthqa_gold"
    # "synthqa_val"
    # "docmatheval" 
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
