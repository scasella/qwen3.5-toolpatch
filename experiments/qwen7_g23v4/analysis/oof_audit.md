| circuit | fit items | held-out set | ambiguous -> NEED_CLARIFICATION (replay argmax) | non-ambiguous argmax kept | stage-1 act: ambiguous ends / max elsewhere | alpha6 at clause end (ambiguous) | J3 act: ambiguous / max elsewhere |
|---|---|---|---|---|---|---|---|
| oof_D1D2 | D1+D2 (128), k1 = 4.0 | D3 | 27/28 | 36/36 | 2.5..749.3 / 0.483 | 0.080..0.999 | 0.0..113.2 / 0.0003 |
| oof_D1D2 | D1+D2 (128), k1 = 4.0 | stress | 28/32 | 32/32 | 0.2..792.0 / 0.386 | 0.000..1.000 | 0.0..110.9 / 0.0003 |
| oof_D1D2_k1rule | D1+D2 (128), k1 = 1.53 | D3 | 27/28 | 36/36 | 0.3..109.1 / 0.447 | 0.000..0.999 | 0.0..105.6 / 0.0003 |
| oof_D1D2_k1rule | D1+D2 (128), k1 = 1.53 | stress | 24/32 | 32/32 | 0.2..115.3 / 0.477 | 0.000..0.999 | 0.0..100.1 / 0.0003 |
| shipped | D1+D2+D3 (192), k1 = 4 | D3 | 28/28 | 36/36 | 57.8..409.9 / 0.016 | 0.992..0.999 | 67.4..109.2 / 0.0003 |
| shipped | D1+D2+D3 (192), k1 = 4 | stress | 27/32 | 32/32 | 0.0..450.1 / 0.012 | 0.000..0.999 | 0.0..106.1 / 0.0003 |

Stage-1 gate fit on D1+D2 only (theta1 = 9.6; gain-independent): pre-activation z at held-out ambiguous clause ends vs every other held-out position:
- oof_D1D2:D3: ambiguous ends [0.43, 6.84] (28/28 above threshold), other positions max 0.25 (61/64 items fully below), margin +0.18
- oof_D1D2:stress: ambiguous ends [-1.10, 7.04] (30/32 above threshold), other positions max 1.17 (57/64 items fully below), margin -2.27

Simulator check (shipped g23v4 replay vs real edited model, pass c01, 64 D1 items): argmax agreement 64/64, |logit error| on NE/NO/<tool_call> median 0.062, max 0.500.
