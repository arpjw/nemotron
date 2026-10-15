# BANKING77 development error analysis

This read-only analysis uses `dev-comparison.json`, its recorded outcomes, and frozen `data/frozen/dev.jsonl` only. No official-test predictions or labels were accessed. It does not change the selected model, decoding, data, or gate. Results describe one seed, one epoch, and this 1,993-example development split.

## Overall change

| Metric | Untouched Nemotron | Selected adapter |
|---|---:|---:|
| Constrained macro-F1 | 0.39915 | 0.82978 |
| Constrained accuracy | 813/1,993 (40.79%) | 1,655/1,993 (83.04%) |
| Constrained malformed outputs | 0 | 0 |
| Raw accuracy | 721/1,993 (36.18%) | 1,655/1,993 (83.04%) |
| Raw malformed outputs | 271 (13.60%) | 2 (0.10%) |
| Raw macro-F1 | 0.37012 | 0.83057 |

The primary macro-F1 gain is **43.06 percentage points**, with paired group-bootstrap 95% interval **[41.00, 45.46] points** (2,000 draws). Accuracy improves **42.25 points**, interval **[39.89, 44.56]**. These are development sampling intervals, not uncertainty across training seeds or domains. The prospective development gate passed.

The primary gain survives identical taxonomy-constrained decoding for both models, where neither has malformed outputs. Raw-format improvement is additional: 269 fewer malformed outputs. Of the baseline's 271 malformed rows, 224 become correct raw adapter predictions. Equal aggregate raw and constrained adapter accuracy does not imply identical row-level predictions.

## Error overlap

| Untouched result | Adapter result | Examples |
|---|---|---:|
| Wrong | Correct | 885 |
| Correct | Correct | 770 |
| Wrong | Wrong | 295 |
| Correct | Wrong | 43 |

The adapter fixes 885/1,180 (75.0%) baseline errors and introduces errors on 43/813 (5.3%) previously correct rows. The remaining 338 errors include 295 shared errors and 43 regressions.

## Class changes and residual weaknesses

All **77 classes improve in F1**; none tie or regress in F1. Class estimates have small supports and were inspected after evaluation, so these rankings are descriptive.

| Largest F1 gains | Support | Base F1 | Adapter F1 |
|---|---:|---:|---:|
| `getting_spare_card` | 26 | 0.000 | 0.962 |
| `activate_my_card` | 32 | 0.061 | 0.969 |
| `exchange_via_app` | 24 | 0.000 | 0.868 |
| `apple_pay_or_google_pay` | 25 | 0.077 | 0.939 |
| `order_physical_card` | 24 | 0.000 | 0.857 |

F1 improvement can coexist with worse class recall. Correct counts fall for `exchange_rate` (22→20 of22), `card_not_working` (15→14 of22), and `verify_my_identity` (21→19 of21). Their F1 still rises because precision improves.

The weakest adapter classes by F1 are `top_up_by_bank_transfer_charge` (0.414;6/22 correct), `topping_up_by_card` (0.533;8/21), `top_up_reverted` (0.571;14/28), `reverted_card_payment?` (0.578;13/32), and `supported_cards_and_currencies` (0.595;11/26).

Frequent remaining confusions include:

- `top_up_by_bank_transfer_charge` → `transfer_fee_charged`: 12 rows.
- `cash_withdrawal_not_recognised` → `pending_cash_withdrawal`: 10 rows.
- `top_up_reverted` → `top_up_failed`: 10 rows.
- `card_payment_wrong_exchange_rate` → `exchange_rate`: 8 rows.
- `direct_debit_payment_not_recognised` → `card_payment_not_recognised`: 8 rows.

These support a narrow interpretation: improved mapping to banking support intents, with persistent confusion between neighboring transaction types and statuses.

## Illustrative development examples

Examples below were selected after seeing outcomes to illustrate specific patterns; they are not a representative sample. ID prefixes identify the frozen rows.

| ID prefix | Public text | Reference | Base → adapter |
|---|---|---|---|
| `c6624895dd7e` | What is the procedure for activating this card? | `activate_my_card` | `card_acceptance` → `activate_my_card` |
| `3828996aad30` | Can you tell me how many currencies I can hold? | `fiat_currency_support` | `supported_cards_and_currencies` → `fiat_currency_support` |
| `80be3b30c4c2` | The top-up did not go through. | `top_up_failed` | `top_up_failed` → `pending_top_up` |
| `18cbf4050b79` | The app says I made a cash withdrawal even though I didn't. | `cash_withdrawal_not_recognised` | `cash_withdrawal_not_recognised` → `pending_cash_withdrawal` |
| `4dd8c24ddd2c` | Is there a fee to make a transfer? | `top_up_by_bank_transfer_charge` | Both: `transfer_fee_charged` |

Benchmark-label agreement is not always unambiguous semantic correctness. For example, row `5128f15deaef` asks “Why has my deposit in the ATM not cleared yet?” but carries `pending_cash_withdrawal`; the baseline chose `balance_not_updated_after_cheque_or_cash_deposit` and the adapter matched the benchmark label. This does not invalidate the benchmark, but cautions against describing every measured correction as better financial understanding. No labels were changed.

## Interpretation

The development evidence supports a substantial improvement over untouched Nemotron on this fixed 77-intent classification task. It does not establish broad finance skill, calibrated confidence, out-of-taxonomy rejection, or production readiness. The official comparison must independently assess the already-selected checkpoint; this analysis supplies no new model-selection decision or test-driven adjustment.
