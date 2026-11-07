

\# Evaluating the Robustness of AI-Generated Text Detection Across Domains and Language Models



A Comparative Study of Detection Performance and Generalization.



\## Overview



This project investigates the robustness of AI-generated text detection models when evaluated on unseen writing domains and unseen language-model families.



It compares traditional machine-learning approaches with a pretrained transformer and investigates whether domain diversity and different feature representations can improve generalization.



\## Research Question



How does the performance of AI-generated-text detection models change when evaluated on unseen text domains and language models, and can lightweight adaptation techniques improve their generalization?



\## Models



\### Traditional baseline

\- Word-level TF-IDF

\- Character-level TF-IDF

\- Combined TF-IDF

\- Logistic Regression



\### Transformer

\- DistilBERT-base-uncased

\- Fine-tuned using the MAGE dataset

\- PyTorch and Hugging Face Transformers



\## Dataset



MAGE (Machine-Generated Text Detection Benchmark).



Labels:

\- 0: AI-generated

\- 1: Human-written



Dataset source:

https://huggingface.co/datasets/yaful/MAGE



The dataset is not included in this repository.



\## Experiments



1\. Dataset auditing and preparation.

2\. Cross-domain robustness evaluation.

3\. Cross-model-family evaluation.

4\. Feature ablation.

5\. Domain adaptation experiments.

6\. Source-bias auditing.

7\. Generator-family fingerprint analysis.

8\. Bootstrap confidence interval analysis.

9\. Paired statistical comparisons.

10\. DistilBERT fine-tuning and evaluation.



\## Technology Stack



\- Python

\- PyTorch

\- Hugging Face Transformers

\- Scikit-learn

\- FastAPI

\- Next.js

\- React

\- TypeScript

\- Tailwind CSS



\## Project Structure



```text

VIT Final Project/

├── frontend/

├── app.py

├── experiments and research scripts

├── outputs/        (excluded from Git)

├── .venv/          (excluded from Git)

├── .gitignore

└── README.md

```



The existing research scripts are maintained in the project root to preserve their original paths.



\## Application



The research application uses a fine-tuned DistilBERT model through a FastAPI backend and a Next.js frontend.



The backend performs model inference and returns a classification estimate with AI and human scores.



The frontend communicates with the backend through HTTP requests.



\## Important Limitations



\- AI-generated-text detection is not definitive proof of authorship.

\- Performance can degrade on unseen domains and generator families.

\- Source metadata in the MAGE dataset presents potential confounding.

\- Some transformer experiments use a single training epoch.

\- The application is a research prototype.



\## Model Checkpoints



Trained model weights are not stored in this Git repository because of their size.



\## Project Status



Machine-learning experiments completed.



FastAPI inference API implemented and tested locally.



Next.js frontend development and application integration in progress.



