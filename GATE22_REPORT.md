# N-MIND Language V2 Gate-2.2 report

## Result

The newly authorized independent round completed all 1,056 optimizer steps within the 90-minute global wall-clock limit. The corrected trainer used interleaved L0/L1 sampling and `answer_span_plus_eos` loss. Main and label-permutation arms started from the identical parameter hash; only the answer permutation differed. The lr=0 arm used the same initial hash and kept parameters bitwise unchanged.

This is `COMPLETED_FAIL` for transferable language binding. The main arm shows a surface learning signal, but it does not beat the independently permuted-label control and passes 0/80 counterfactual pairs. The requested question—whether N-Cell learned transferable unseen-situation language ability—is answered: **not demonstrated in this round**.

## Budget and training

| arm | steps | answer loss | parameter change | wall seconds |
|---|---:|---:|---|---:|
| main | 512 | 5.499861 → 0.722144 | yes | 183.75 |
| label permutation | 512 | 5.633407 → 1.014056 | yes | 183.11 |
| lr=0 | 32 | 5.499861 → 5.596822 | no, bitwise | 11.44 |

Total: 1,056 steps, 533.29 seconds, CPU. Every run used 512-row interleaved windows containing L0=256 and L1=256. Stable was read-only and no promotion occurred.

## Neural-only sealed evaluation

The same 240 sealed items and 80 counterfactual pairs were used with the same greedy continuation protocol. Memory, retrieval, tools, rules and external models were disabled.

| arm | sealed exact-match | L0 | L1 | counterfactual pairs |
|---|---:|---:|---:|---:|
| zero random baseline | 0.00% | 0.00% | 0.00% | 0/80 |
| main | 16.67% | 0.00% | 25.00% | 0/80 |
| label permutation | 16.67% | 0.00% | 25.00% | 0/80 |
| lr=0 | 0.00% | 0.00% | 0.00% | 0/80 |

The main-minus-permutation gain is 0 percentage points. Therefore this result is `SURFACE_LANGUAGE_ONLY_BINDING_FAIL` and also `CONTROL_NOT_SEPARATED`; it is not a validated neural skill.

## Interpretation

The answer-span mask fixed the previous prompt-copy loss problem and the balanced sampler ensured L1 was actually seen. Loss fell and the main arm produced some L1 answers, but the identical result from the label-permutation arm shows the remaining signal can be explained by label/token distribution or output-surface shortcuts. The 0/80 pair result rules out the required entity-binding claim. No claim is made about multi-turn, reference resolution or childlike language ability.

## Artifacts and commands

- `experiments/language_v2/gate22/GATE22_PREREGISTRATION.json`
- `experiments/language_v2/gate22/GATE22_PREFLIGHT_VERIFIED.json`
- `experiments/language_v2/gate22/GATE22_TRAIN_MAIN.json`
- `experiments/language_v2/gate22/GATE22_TRAIN_PERMUTED.json`
- `experiments/language_v2/gate22/GATE22_TRAIN_LR0.json`
- `experiments/language_v2/gate22/GATE22_EVAL_PER_ITEM.json`
- `experiments/language_v2/gate22/GATE22_BUDGET_AUDIT.json`

```powershell
$env:PYTHONPATH = "src"
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" experiments\language_v2\gate22\run_gate22.py
```

The command is intentionally not to be rerun without a new authorization; it would consume another training round.
