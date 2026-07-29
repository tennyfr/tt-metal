# skillexp — machine a

generated 2026-07-29T17:43:23+00:00 on qb2-120-p05t03

tt-metal HEAD `78dbd88bec7` on
`HEAD`

## Phase 1 — functional decoder (owned by this machine)

| model | goal | gate | fd-ready tag |
|---|---|---|---|
| `microsoft_phi_3_5_mini_instruct` | complete | - | yes |
| `qwen_qwen3_6_27b` | complete | - | yes |
| `coherelabs_north_mini_code_1_0` | not-started | - | yes |
| `google_gemma_4_26b_a4b_it` | not-started | - | yes |

## Phase 2/3 — optimize, this machine's arms

| arm | model | goal | gate | done tag |
|---|---|---|---|---|
| nofuse-advise | `microsoft_phi_3_5_mini_instruct` | complete | pass | yes |
| nofuse-advise | `qwen_qwen3_6_27b` | complete | pass | yes |
| nofuse-advise | `coherelabs_north_mini_code_1_0` | complete | pass | yes |
| nofuse-advise | `google_gemma_4_26b_a4b_it` | running?-no-multigoal-proc | - | no |
| fuse-advise | `microsoft_phi_3_5_mini_instruct` | complete | pass | yes |
| fuse-advise | `qwen_qwen3_6_27b` | running?-no-multigoal-proc | - | no |
| fuse-advise | `coherelabs_north_mini_code_1_0` | running?-no-multigoal-proc | - | no |
| fuse-advise | `google_gemma_4_26b_a4b_it` | running?-no-multigoal-proc | - | no |

## Device
```
tt-smi suppressed: measured stage may be live
```

## Blocked / critical (audit these before calling them real blockers)
- none
