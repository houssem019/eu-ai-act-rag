import os

# Ragas posts anonymous usage analytics to its vendor by default. Opt out before it is ever imported: evals run
# in CI and on shared machines, and nothing should phone home without being asked to. Override by setting the
# variable explicitly.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
