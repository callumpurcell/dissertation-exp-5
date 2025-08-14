export CUDA_VISIBLE_DEVICES=4,5,6,7
echo "Running: llama32_1B_num_60_control_1200_2e-4_20250813_211405"
axolotl train configs/llama32_1B_num_60_control_1200_2e-4_20250813_211405.yaml --accelerate
echo "Running: llama32_1B_agreement-2-models_1200_2e-4_20250813_211405"
axolotl train configs/llama32_1B_agreement-2-models_1200_2e-4_20250813_211405.yaml --accelerate
echo "Running: llama32_1B_mistral_3B_1200_2e-4_20250813_211405"
axolotl train configs/llama32_1B_mistral_3B_1200_2e-4_20250813_211405.yaml --accelerate
echo "Running: llama32_1B_mistral_8B_1200_2e-4_20250813_211405"
axolotl train configs/llama32_1B_mistral_8B_1200_2e-4_20250813_211405.yaml --accelerate
echo "Running: llama32_1B_mistral_medium_1200_2e-4_20250813_211405"
axolotl train configs/llama32_1B_mistral_medium_1200_2e-4_20250813_211405.yaml --accelerate
echo "Running: llama32_1B_mistral_small_1200_2e-4_20250813_211405"
axolotl train configs/llama32_1B_mistral_small_1200_2e-4_20250813_211405.yaml --accelerate
