# Claude 3 Model Card

## Model Overview

### Model Family
The Claude 3 Model Family: Opus, Sonnet, Haiku

### Publisher
Anthropic

### Model Variants
- **Claude 3 Opus**: Most capable offering with state-of-the-art performance.
- **Claude 3 Sonnet**: Balances skills and speed.
- **Claude 3 Haiku**: Fastest and most affordable model.

### Capabilities
All models have vision capabilities enabling them to process and analyze image data. They excel in reasoning, math, coding, and fluency in non-English languages. Claude 3 Opus achieves state-of-the-art results in evaluations such as GPQA, MMLU, and MMMU. Claude 3 Haiku performs comparably to Claude 2 on most text tasks, while Sonnet and Opus outperform it significantly.

## Introduction
The Claude 3 family of models sets new benchmarks in reasoning, math, coding, multilingual understanding, and vision quality. They are trained using various methods, including unsupervised learning and Constitutional AI. The models are developed using AWS and GCP hardware and frameworks like PyTorch, JAX, and Triton. Key enhancements include multimodal input capabilities and advanced tool use for integration into specialized applications.

## Model Details

### Intended Uses
Claude models are designed to be helpful, honest, and harmless assistants. They excel at open-ended conversation, collaboration on ideas, and coding tasks. Their multimodal features support visual input interpretation for diverse use cases, making them adaptive and engaging.

### Unintended Uses
The models should not be used independently in high-stakes situations where incorrect answers could cause harm. They should not replace professionals like lawyers or doctors and should not be deployed without human oversight. Claude models do not have real-time web search capabilities.

### Prohibited Uses
According to Anthropic's Acceptable Use Policy (AUP), prohibited uses include political campaigning, surveillance, social scoring, criminal justice decisions, law enforcement, and decisions related to financing, employment, and housing. The AUP requires disclosure of AI use and outlines human-in-the-loop measures for certain cases.

### Safeguarding Against Misuse
Anthropic employs automated systems to detect and mitigate prohibited uses in real-time. Prompts violating the AUP trigger cautious responses or blocking of the model's output. Repeat violations can result in termination of access.

### Training Data
Claude 3 models are trained on publicly available information up to August 2023, non-public third-party data, data from labeling services, and internally generated data. They do not use user-submitted prompt or output data for training. Data crawling follows industry practices and respects website permissions.

### Training Process
The models are trained to be helpful, harmless, and honest, using techniques like word prediction and human feedback. Constitutional AI aligns the models with human values, including respect for disability rights. Continuous evaluations and classifiers monitor safety and alignment with the AUP.

### Release Decisions and Maintenance
Anthropic follows the NIST AI Risk Management Framework for responsible AI development and deployment. This includes documenting permissible uses, evaluating system performance and safety, and rolling out access incrementally. Data privacy is prioritized, and a Responsible Scaling Policy guides development and deployment practices.

## Security
The security measures include connection authentication, multi-factor authentication, two-party controls, continuous monitoring, endpoint hardening, and penetration testing. Access to model infrastructure is strictly controlled and monitored.

## Social Responsibility

### Constitutional AI
Claude models are guided by a set of ethical principles to avoid harmful outputs and unethical activities. The principles include respect for human rights and accessibility for individuals with disabilities.

### Labor
Anthropic works with data platforms to manage tasks like model output selection, evaluation, and adversarial testing. This data work supports technical safety research and model training.

### Sustainability
Anthropic offsets its operational carbon emissions and partners with cloud providers that prioritize renewable energy. The goal is to maintain net-zero climate impact through verified carbon credits and emissions reduction projects.

## Core Capabilities Evaluations
The Claude 3 family has been evaluated across various domains, including reasoning, multilingual tasks, long context handling, honesty/factuality, and multimodal capabilities. These evaluations use industry-standard benchmarks and additional internal benchmarks for harmless refusals.

### Benchmarks and Performance
Claude 3 models demonstrate superior capabilities in reasoning, reading comprehension, math, science, and coding, achieving state-of-the-art results in many cases. Evaluations include GPQA, MMLU, ARC-Challenge, PubMedQA, GSM8K, MATH, MGSM, HellaSwag, WinoGrande, DROP, RACE-H, QuALITY, HumanEval, APPS, MBPP, and BIG-Bench-Hard.

## Contact and Further Information
For comprehensive insights into training and evaluation methodologies, refer to Anthropic's research papers and documentation.