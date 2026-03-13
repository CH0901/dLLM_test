## Evaluation

Archived `dream-7b-instruct` results reproduced with the QDLM evaluation flow:

### Code

| Model | Task | Metric | Score |
| --- | --- | --- | ---: |
| dream-7b-instruct FP16 | `humaneval_instruct_noprefix` | `pass@1` | 0.5671 |
| dream-7b-instruct AWQ | `humaneval_instruct_noprefix` | `pass@1` | 0.5061 |
| dream-7b-instruct GPTQ | `humaneval_instruct_noprefix` | `pass@1` | 0.4268 |

### Math

| Model | Task | Metric | Score |
| --- | --- | --- | ---: |
| dream-7b-instruct FP16 | `gsm8k` | `exact_match, flexible-extract` | 0.7983 |
| dream-7b-instruct AWQ | `gsm8k` | `exact_match, flexible-extract` | 0.7908 |
| dream-7b-instruct GPTQ | `gsm8k` | `exact_match, flexible-extract` | 0.7604 |
