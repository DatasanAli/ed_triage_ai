# ED Triage AI

Production-ready clinical decision support system for emergency department triage using fine-tuned language models and multi-layered explainability frameworks.

## Overview

Emergency department (ED) triage is a critical decision point in healthcare delivery, requiring rapid assessment and prioritization of patients based on clinical urgency. This system provides AI-powered decision support to enhance triage accuracy while maintaining the transparency and interpretability required for clinical adoption.

## Problem Statement

Traditional ML approaches to triage prediction achieve 65-75% accuracy but function as "black boxes," providing classifications without clinical reasoning or supporting evidence. This lack of interpretability prevents adoption in safety-critical healthcare settings where clinicians require transparent, evidence-based decision support.

## Solution

A multi-component clinical decision support system that combines:

- **Accurate ESI Prediction**: DL models fine-tuned on emergency department clinical text and structured data
- **Multi-Layered Explainability**:
  - SHAP analysis for feature attribution
  - Retrieval-augmented generation (RAG) for evidence from similar historical cases
  - LLM synthesis for clinically coherent reasoning
- **Interactive Interface**: Web application designed for ED triage nurses and physicians

## Key Features

- **ESI Level Prediction**: Five-level acuity classification (Level 1: immediate life-saving intervention → Level 5: non-urgent)
- **Confidence Scoring**: Transparent uncertainty quantification for each prediction
- **Feature Attribution**: Highlights which clinical features drive each prediction
- **Case Retrieval**: Surface similar historical cases with documented outcomes
- **Evidence-Grounded Reasoning**: Generate clinically coherent explanations for triage decisions

## Architecture

The system integrates three complementary explainability layers:

1. **Model Interpretability**: SHAP values for feature-level attribution
2. **Evidence Retrieval**: RAG pipeline for similar case identification
3. **Clinical Reasoning**: LLM-generated synthesis of evidence and prediction rationale

## Design Philosophy

This system is designed to **support, not replace**, clinical judgment by providing transparent, actionable decision support that clinicians can integrate into existing triage workflows.

## Technical Stack

- Fine-tuned DL models for classification
- SHAP for model interpretability
- RAG pipeline for evidence retrieval
- LLM integration for reasoning synthesis
- Interactive web interface

## Dataset

Development utilizes de-identified emergency department encounter data including:
- Ground-truth ESI labels
- Chief complaints
- Vital signs
- Demographics

In production deployment, equivalent data would be extracted in real-time from hospital electronic health record systems during the triage process.

## Deployment

The model is deployed via an automated SageMaker Pipeline. See [sagemaker/README.md](sagemaker/README.md) for pipeline documentation, deployment instructions, and endpoint testing.

## Use Cases

- **Emergency Department Triage**: Primary decision support for triage nurses
- **Clinical Training**: Educational tool for triage decision-making
- **Quality Assurance**: Retrospective analysis of triage decisions
- **Care Gap Identification**: Detection of potential delays in care delivery

## Project Status

This is a capstone project demonstrating production-ready clinical decision support system design and implementation.

## License

## References

- Gilboy, N., et al. (2011). Emergency Severity Index (ESI): A Triage Tool for Emergency Department Care
- Levin, S., et al. (2018). Machine-learning-based electronic triage
- Raita, Y., et al. (2019). Emergency department triage prediction of clinical outcomes
- Cutillo, C. M., et al. (2024). Machine intelligence in healthcare

---

**Note**: This system is intended for research and educational purposes. Clinical deployment requires appropriate validation, regulatory approval, and integration with hospital workflows.
