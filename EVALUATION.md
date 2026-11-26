# Prototype Evaluation Guide

## Automated tests

Run:

```powershell
python -m unittest -v
```

The test suite evaluates document chunk boundaries, retrieval relevance, grounding flags, malformed-model
output handling, Ollama error reporting, runtime prompt consistency, and fine-tuning data validity.

## Human evaluation

Use anonymised blockchain and digital-asset case studies: DeFi lending/liquidation, wallet custody and key recovery,
stablecoin issuance/reserves, cross-chain payments, tokenised trade finance, KYC/AML and CBDC. Record both an expert
baseline and the project-system measurements:

| Metric | Method |
|---|---|
| Precision / recall / F1 | Compare extracted requirements with an expert-labelled gold set |
| Completeness | Expert review against mandatory project and stakeholder categories |
| Ambiguity and conflict accuracy | Compare detector findings with independent reviewers |
| Regulatory-control coverage | Map outputs to an approved control catalogue |
| Citation correctness | Verify each citation against the retrieved source and effective version |
| Hallucination rate | Count unsupported claims divided by total generated claims |
| SDLC suitability | Compare ranked recommendations with architects and project managers |
| Inferred project-factor quality | Compare each low/medium/high factor and its effect on the SDLC ranking with expert labels |
| Human correction rate | Count requirement or SDLC changes made during approval |
| Processing time | Compare end-to-end elapsed time with a conventional baseline |
| Stakeholder satisfaction | Use an anonymised structured survey after completing a case study |
| Traceability coverage | Percentage of requirements linked to source, artefacts and validation evidence |
| Smart-contract control coverage | Expert-checked coverage of access control, invariants, upgrade/pause controls, oracle/bridge handling and transaction-integrity tests |
| On-chain/off-chain privacy quality | Expert-checked data placement, minimisation and retention decisions |

## Interpretation

Report sample size, uncertainty intervals, project chain/environment, source versions and reviewer qualifications.
The sample corpus is synthetic, not authoritative regulation; do not claim regulatory accuracy without an
organisation-approved, versioned knowledge base and legal/compliance review.
The bundled factor labels are illustrative heuristics generated from synthetic seed narratives, not expert
annotations. Evaluate and replace them with reviewed examples before relying on fine-tuning.
