Answerable questions: 55. Hit = any gold citation in the top-k chunks. ctx@5 = mean tokens in the top-5 chunks. Latency is per query on CPU (laptop), including query embedding.

| Chunking | Embedder | Mode | R@1 | R@5 | R@10 | MRR@10 | ctx@5 tokens | p50 ms |
|---|---|---|---|---|---|---|---|---|
| paragraph | bge-small | rerank | 0.73 | 0.95 | 0.98 | 0.834 | 887 | 5034 |

Refusal threshold on the top rerank score (55 answerable vs 8 unanswerable questions):

| Chunking | Embedder | Threshold | Answer rate (answerable) | Refusal rate (unanswerable) |
|---|---|---|---|---|
| paragraph | bge-small | 1.37 | 0.98 | 0.88 |
