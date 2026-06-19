# Argus evaluation metrics (sample_claims.csv)

## Per-field accuracy

| metric | rules |
|---|---|
| evidence_standard_met | 0.750 |
| valid_image | 0.900 |
| issue_type | 0.500 |
| object_part | 0.800 |
| claim_status | 0.800 |
| severity | 0.450 |
| claim_status_macro_f1 | 0.689 |
| risk_flags_f1 | 0.769 |
| supporting_image_ids_f1 | 0.769 |

**Final strategy chosen for output.csv:** `rules`

Confusion matrix (rows = expected, cols = predicted) for the final strategy:

| expected \ predicted | supported | contradicted | not_enough_information |
|---|---|---|---|
| supported | 12 | 0 | 1 |
| contradicted | 1 | 2 | 2 |
| not_enough_information | 0 | 0 | 2 |
