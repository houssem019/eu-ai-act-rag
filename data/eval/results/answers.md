Model: `qwen2.5:7b`. 55 answerable + 8 unanswerable questions.

| Answer rate | Refusal rate | Citation accuracy | Invalid citations (draft) | Cites article/annex | p50 latency | Cost/question |
|---|---|---|---|---|---|---|
| 0.96 | 1.00 | 0.92 | 0.02 | 0.96 | 74.8s | $0.0000 |

Refusals by trigger: {'low_evidence': 8, 'model': 2, 'uncited': 0, 'safety': 0}

| id | refused | cited | gold hit |
|---|---|---|---|
| d01 |  | Article 3(1) | yes |
| d02 |  | Article 3(4) | yes |
| d03 |  | Article 3(60) | yes |
| d04 |  | Article 3(49) | yes |
| d05 |  | Article 3(63) | yes |
| d06 |  | Article 3(23) | yes |
| p01 |  | Article 5(1) | yes |
| p02 |  | Article 5(1) | yes |
| p03 |  | Recital 44 | **no** |
| p04 |  | Article 5(2); Article 5(3) | yes |
| p05 |  | Article 5(3) | yes |
| s01 |  | Article 2(3) | yes |
| s02 |  | Article 2(6); Recital 25; Recital 97 | yes |
| s03 |  | Article 2(10) | yes |
| s04 |  | Article 54(1); Article 22(1) | **no** |
| s05 |  | Article 2(12); Article 54(6); Article 53(2) | yes |
| h01 |  | Article 6(2) | yes |
| h02 |  | Article 6(3) | yes |
| h03 |  | Annex III | yes |
| h04 |  | Annex III; Article 6(3) | yes |
| h05 |  | Annex III | yes |
| h06 |  | Article 10(3); Article 10(2) | yes |
| h07 |  | Article 12(1); Article 12(3); Article 12(2); Article 19(1); Article 26(6) | yes |
| h08 |  | Article 14(1); Article 14(3); Article 14(5); Article 14(4) | yes |
| h09 | model |  |  |
| h10 |  | Article 17(1) | yes |
| h11 |  | Article 22(1) | yes |
| h12 |  | Article 25(1) | yes |
| h13 |  | Article 43(2) | yes |
| h14 |  | Article 43(4) | yes |
| h15 |  | Article 48(2) | yes |
| h16 |  | Article 49(1) | yes |
| dp01 |  | Article 26(6) | yes |
| dp02 |  | Article 26(7) | yes |
| dp03 |  | Article 27(3); Recital 96 | **no** |
| dp04 |  | Article 86(1) | yes |
| t01 | low_evidence |  |  |
| t02 |  | Article 50(2) | yes |
| t03 |  | Article 50(4) | yes |
| g01 |  | Article 53(1); Article 53(3); Article 55(1); Article 53(4) | yes |
| g02 |  | Article 51(2) | yes |
| g03 |  | Article 55(1) | yes |
| g04 |  | Article 52(1) | yes |
| g05 |  | Article 53(2) | yes |
| g06 |  | Article 101(1) | yes |
| e01 |  | Article 99(3) | yes |
| e02 |  | Article 99(5) | yes |
| e03 |  | Recital 168 | **no** |
| e04 |  | Article 73(4); Article 73(2) | yes |
| e05 |  | Article 73(4) | yes |
| e06 |  | Article 57(1) | yes |
| tl01 |  | Article 113 | yes |
| tl02 |  | Article 113 | yes |
| tl03 |  | Article 111(3) | yes |
| l01 |  | Article 4; Article 95(2); Article 66 | yes |
| u01 | low_evidence |  |  |
| u02 | low_evidence |  |  |
| u03 | low_evidence |  |  |
| u04 | model |  |  |
| u05 | low_evidence |  |  |
| u06 | low_evidence |  |  |
| u07 | low_evidence |  |  |
| u08 | low_evidence |  |  |
