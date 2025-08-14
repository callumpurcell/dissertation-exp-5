# SynthQA Evaluation Task

This task evaluates models on the SynthQA dataset, which contains question-answer pairs about various topics including business, finance, and general knowledge.

## Dataset Format

The dataset is in JSONL format with each line containing:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Question text..."
    },
    {
      "role": "assistant", 
      "content": "Answer (number, text, or boolean)"
    }
  ]
}
```

## Evaluation Metrics

1. **equal_value_custom**: Custom evaluation that handles:
   - Numerical answers with tolerance (e.g., 9.47 ≈ 9.47)
   - Boolean answers (Yes/No, True/False)
   - Currency conversions ($5 million → 5000000)
   - Percentage conversions (55.5% → 0.555)
   - Text normalization

2. **equal_math_verify**: Uses the `math_verify` library for mathematical expression evaluation

## Usage

To evaluate a model on this task:

```bash
# Evaluate on validation set
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct \
  --tasks synthqa \
  --batch_size 1 \
  --output_path results.json

# Use a fine-tuned model
lm_eval --model hf \
  --model_args pretrained=./outputs/your_model \
  --tasks synthqa \
  --batch_size 1 \
  --output_path results.json
```

## Files

- `synthqa.yaml`: Task configuration
- `utils.py`: Answer extraction and comparison functions
- `all.yaml`: Task group definition

## Notes

- The task automatically extracts answers from generated text
- Supports various answer formats (numbers, percentages, currency, booleans)
- Uses relative tolerance for floating-point comparisons 