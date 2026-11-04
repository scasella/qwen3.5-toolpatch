| cohort | model | lenient | strict | false calls | per class (correct/n, decisions) | mean tokens | vs Base: w->r / r->w / identical | vs g23v3: w->r / r->w / identical |
|---|---|---|---|---|---|---|---|---|
| Practice (64) | Base | 24/64 | 7/64 | 38 | valid_call:24/24 {'call': 24}, no_tool:0/20 {'call': 18, 'prose': 2}, need_clarification:0/10 {'call': 10}, ambiguous:0/10 {'call': 10} | 55.8 | - | 0 / 28 / 36 |
| Practice (64) | g23v3 | 52/64 | 35/64 | 12 | valid_call:24/24 {'call': 24}, no_tool:19/20 {'answer': 19, 'call': 1}, need_clarification:9/10 {'clarify': 9, 'call': 1}, ambiguous:0/10 {'call': 10} | 33.5 | 28 / 0 / 36 | - |
| Practice (64) | g23v4 | 62/64 | 45/64 | 2 | valid_call:24/24 {'call': 24}, no_tool:19/20 {'answer': 19, 'call': 1}, need_clarification:9/10 {'clarify': 9, 'call': 1}, ambiguous:10/10 {'clarify': 10} | 24.7 | 38 / 0 / 26 | 10 / 0 / 54 |
| Selection (32) | Base | 12/32 | 4/32 | 17 | valid_call:12/12 {'call': 12}, no_tool:0/10 {'call': 7, 'prose': 3}, need_clarification:0/5 {'call': 5}, ambiguous:0/5 {'call': 5} | 56.8 | - | 0 / 15 / 17 |
| Selection (32) | g23v3 | 27/32 | 19/32 | 5 | valid_call:12/12 {'call': 12}, no_tool:10/10 {'answer': 10}, need_clarification:5/5 {'clarify': 5}, ambiguous:0/5 {'call': 5} | 34.0 | 15 / 0 / 17 | - |
| Selection (32) | g23v4 | 32/32 | 24/32 | 0 | valid_call:12/12 {'call': 12}, no_tool:10/10 {'answer': 10}, need_clarification:5/5 {'clarify': 5}, ambiguous:5/5 {'clarify': 5} | 24.1 | 20 / 0 / 12 | 5 / 0 / 27 |
| Renamed A (32) | Base | 8/32 | 6/32 | 20 | valid_call:8/12 {'call': 12}, no_tool:0/10 {'call': 10}, need_clarification:0/5 {'call': 5}, ambiguous:0/5 {'call': 5} | 45.8 | - | 0 / 15 / 17 |
| Renamed A (32) | g23v3 | 23/32 | 21/32 | 5 | valid_call:8/12 {'call': 12}, no_tool:10/10 {'answer': 10}, need_clarification:5/5 {'clarify': 5}, ambiguous:0/5 {'call': 5} | 24.5 | 15 / 0 / 17 | - |
| Renamed A (32) | g23v4 | 28/32 | 26/32 | 0 | valid_call:8/12 {'call': 12}, no_tool:10/10 {'answer': 10}, need_clarification:5/5 {'clarify': 5}, ambiguous:5/5 {'clarify': 5} | 18.8 | 20 / 0 / 12 | 5 / 0 / 27 |
| Renamed B (32) | Base | 12/32 | 10/32 | 20 | valid_call:12/12 {'call': 12}, no_tool:0/10 {'call': 10}, need_clarification:0/5 {'call': 5}, ambiguous:0/5 {'call': 5} | 45.9 | - | 0 / 14 / 17 |
| Renamed B (32) | g23v3 | 26/32 | 25/32 | 6 | valid_call:12/12 {'call': 12}, no_tool:10/10 {'answer': 10}, need_clarification:4/5 {'clarify': 4, 'call': 1}, ambiguous:0/5 {'call': 5} | 26.8 | 14 / 0 / 17 | - |
| Renamed B (32) | g23v4 | 31/32 | 30/32 | 1 | valid_call:12/12 {'call': 12}, no_tool:10/10 {'answer': 10}, need_clarification:4/5 {'clarify': 4, 'call': 1}, ambiguous:5/5 {'clarify': 5} | 20.8 | 19 / 0 / 12 | 5 / 0 / 27 |
| Final-split proxy (32) | Base | 12/32 | 3/32 | 19 | valid_call:12/12 {'call': 12}, no_tool:0/10 {'call': 9, 'prose': 1}, need_clarification:0/5 {'call': 5}, ambiguous:0/5 {'call': 5} | 54.7 | - | 0 / 15 / 17 |
| Final-split proxy (32) | g23v3 | 27/32 | 18/32 | 5 | valid_call:12/12 {'call': 12}, no_tool:10/10 {'answer': 10}, need_clarification:5/5 {'clarify': 5}, ambiguous:0/5 {'call': 5} | 32.1 | 15 / 0 / 17 | - |
| Final-split proxy (32) | g23v4 | 32/32 | 23/32 | 0 | valid_call:12/12 {'call': 12}, no_tool:10/10 {'answer': 10}, need_clarification:5/5 {'clarify': 5}, ambiguous:5/5 {'clarify': 5} | 23.6 | 20 / 0 / 12 | 5 / 0 / 27 |
| Short non-tool regression2 (32) | Base | 28/32 | 28/32 | 0 | regression:28/32 {'None': 32} | 8.7 | - | 0 / 0 / 32 |
| Short non-tool regression2 (32) | g23v3 | 28/32 | 28/32 | 0 | regression:28/32 {'None': 32} | 8.7 | 0 / 0 / 32 | - |
| Short non-tool regression2 (32) | g23v4 | 28/32 | 28/32 | 0 | regression:28/32 {'None': 32} | 8.7 | 0 / 0 / 32 | 0 / 0 / 32 |
| Prose QA 128-token (16 derivation set) | Base | 16/16 | 16/16 | 0 | prose:16/16 {'None': 16} | 122.1 | - | 0 / 0 / 11 |
| Prose QA 128-token (16 derivation set) | g23v3 | 16/16 | 16/16 | 0 | prose:16/16 {'None': 16} | 122.1 | 0 / 0 / 11 | - |
| Prose QA 128-token (16 derivation set) | g23v4 | 16/16 | 16/16 | 0 | prose:16/16 {'None': 16} | 122.1 | 0 / 0 / 7 | 0 / 0 / 11 |
| Prose QA 128-token (16 fresh) | Base | 16/16 | 16/16 | 0 | prose:16/16 {'None': 16} | 127.9 | - | 0 / 0 / 9 |
| Prose QA 128-token (16 fresh) | g23v3 | 16/16 | 16/16 | 0 | prose:16/16 {'None': 16} | 128.0 | 0 / 0 / 9 | - |
| Prose QA 128-token (16 fresh) | g23v4 | 16/16 | 16/16 | 0 | prose:16/16 {'None': 16} | 128.0 | 0 / 0 / 8 | 0 / 0 / 12 |
| Stress: ambiguous (32) | Base | 0/32 | 0/32 | 32 | ambiguous:0/32 {'call': 32} | 54.2 | - | 0 / 9 / 16 |
| Stress: ambiguous (32) | g23v3 | 9/32 | 9/32 | 16 | ambiguous:9/32 {'clarify': 9, 'call': 16, 'answer': 6, 'prose': 1} | 32.5 | 9 / 0 / 16 | - |
| Stress: ambiguous (32) | g23v4 | 27/32 | 27/32 | 5 | ambiguous:27/32 {'clarify': 27, 'call': 5} | 13.2 | 27 / 0 / 5 | 18 / 0 / 14 |
| Stress: resolved two-value calls (32) | Base | 30/32 | 15/32 | 0 | resolved:30/32 {'call': 32} | 51.8 | - | 0 / 0 / 32 |
| Stress: resolved two-value calls (32) | g23v3 | 30/32 | 15/32 | 0 | resolved:30/32 {'call': 32} | 51.8 | 0 / 0 / 32 | - |
| Stress: resolved two-value calls (32) | g23v4 | 30/32 | 15/32 | 0 | resolved:30/32 {'call': 32} | 51.8 | 0 / 0 / 32 | 0 / 0 / 32 |
