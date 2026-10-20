Answerable questions: 55. Hit = any gold citation in the top-k chunks. ctx@5 = mean tokens in the top-5 chunks. Latency is per query on CPU (laptop), including query embedding.

| Chunking | Embedder | Mode | R@1 | R@5 | R@10 | MRR@10 | ctx@5 tokens | p50 ms |
|---|---|---|---|---|---|---|---|---|
| article | bge-small | dense | 0.31 | 0.73 | 0.78 | 0.478 | 1342 | 78 |
| article | bge-small | hybrid | 0.29 | 0.78 | 0.87 | 0.509 | 1397 | 141 |
| article | bge-small | rerank | 0.35 | 0.78 | 0.87 | 0.557 | 1353 | 5418 |
| paragraph | bge-small | dense | 0.51 | 0.84 | 0.91 | 0.647 | 875 | 78 |
| paragraph | bge-small | hybrid | 0.35 | 0.87 | 0.91 | 0.559 | 1004 | 167 |
| paragraph | bge-small | rerank | 0.56 | 0.89 | 0.93 | 0.712 | 899 | 5419 |
| fixed | bge-small | dense | 0.33 | 0.64 | 0.71 | 0.464 | 1280 | 77 |
| fixed | bge-small | hybrid | 0.35 | 0.78 | 0.84 | 0.531 | 1277 | 141 |
| fixed | bge-small | rerank | 0.45 | 0.78 | 0.89 | 0.604 | 1280 | 3878 |
| article | bge-base | dense | 0.29 | 0.73 | 0.82 | 0.491 | 1360 | 151 |
| article | bge-base | hybrid | 0.29 | 0.84 | 0.89 | 0.519 | 1417 | 228 |
| article | bge-base | rerank | 0.35 | 0.78 | 0.87 | 0.555 | 1367 | 5813 |
| paragraph | bge-base | dense | 0.53 | 0.91 | 0.95 | 0.689 | 855 | 85 |
| paragraph | bge-base | hybrid | 0.44 | 0.89 | 0.93 | 0.632 | 982 | 163 |
| paragraph | bge-base | rerank | 0.56 | 0.91 | 0.93 | 0.711 | 911 | 3356 |
| fixed | bge-base | dense | 0.35 | 0.69 | 0.76 | 0.481 | 1280 | 109 |
| fixed | bge-base | hybrid | 0.36 | 0.75 | 0.85 | 0.522 | 1277 | 119 |
| fixed | bge-base | rerank | 0.45 | 0.78 | 0.89 | 0.604 | 1280 | 2484 |

Refusal threshold on the top rerank score (55 answerable vs 8 unanswerable questions):

| Chunking | Embedder | Threshold | Answer rate (answerable) | Refusal rate (unanswerable) |
|---|---|---|---|---|
| article | bge-small | 1.78 | 0.93 | 0.88 |
| paragraph | bge-small | 2.36 | 0.96 | 0.88 |
| fixed | bge-small | 2.23 | 0.89 | 0.88 |
| article | bge-base | 1.78 | 0.93 | 0.88 |
| paragraph | bge-base | 2.36 | 0.96 | 0.88 |
| fixed | bge-base | 2.23 | 0.89 | 0.88 |
