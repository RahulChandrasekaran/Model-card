# Llama 3 Model Card

## Model Details

- **Developer:** Meta
- **Model Family:** Meta Llama 3
- **Variations:**
  - **8B:** Pre-trained and instruction-tuned.
  - **70B:** Pre-trained and instruction-tuned.
- **Input:** Text only
- **Output:** Text and code only
- **Architecture:** Optimized transformer architecture utilizing auto-regressive methods.

## Training Overview

- **Training Data:** Mix of publicly available online data.
- **Params:**
  - 8B model: 8 billion
  - 70B model: 70 billion
- **Context Length:** 8k tokens
- **Global QA (GQA):** Yes
- **Token Count:** 15T+
- **Knowledge Cutoff:**
  - 8B model: March 2023
  - 70B model: December 2023
- **Release Date:** April 18, 2024
- **Status:** Static model trained on an offline dataset. Future versions to include community feedback improvements.

## Intended Use

### Use Cases
- **Commercial and Research Use:** English language tasks.
- **Instruction-Tuned Models:** Optimized for assistant-like chat.
- **Pre-Trained Models:** Adaptable for various natural language generation tasks.

### Out-of-Scope Uses
- Violation of laws or regulations.
- Use prohibited by the [Acceptable Use Policy](https://llama.meta.com/llama3/use-policy/) and Llama 3 Community License.
- Non-English languages without compliance with the Community License.

## Benchmarks

- See [Llama 3 documentation](https://github.com/meta-llama/llama3) for detailed benchmark results.

## Ethical Considerations

- **Ethical Values:** Openness, inclusivity, and helpfulness.
- **Risks:** Potential for biased or objectionable responses; recommend safety testing before deployment.

## Resources

- **License:** Custom commercial license available at [Llama 3 License](https://llama.meta.com/llama3/license).
- **Feedback:** Model feedback instructions in the README.
- **Documentation:** [Llama 3 documentation](https://github.com/meta-llama/llama3).

## Citation

- @article{llama3modelcard,
  title={Llama 3 Model Card},
  author={AI@Meta},
  year={2024},
  url={https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md}
}
