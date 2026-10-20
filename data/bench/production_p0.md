Answerable questions: 55. Hit = any gold citation in the top-k chunks. ctx@5 = mean tokens in the top-5 chunks. Latency is per query on CPU (laptop), including query embedding.

| Chunking | Embedder | Mode | R@1 | R@5 | R@10 | MRR@10 | ctx@5 tokens | p50 ms |
|---|---|---|---|---|---|---|---|---|
| paragraph | bge-small | rerank | 0.58 | 0.93 | 0.96 | 0.739 | 897 | 3817 |

Refusal threshold on the top rerank score (55 answerable vs 8 unanswerable questions):

| Chunking | Embedder | Threshold | Answer rate (answerable) | Refusal rate (unanswerable) |
|---|---|---|---|---|
| paragraph | bge-small | 2.04 | 0.96 | 0.88 |
