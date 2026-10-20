"""The Ragas judge bridge, offline: real Ragas metrics driven by a scripted LLM through our client interface."""

import asyncio
import json
import math

import pytest
from ragas.metrics.collections import AnswerCorrectness, Faithfulness
from ragas.metrics.collections.faithfulness.util import StatementGeneratorOutput

from aiact.evals.judge import JudgeError, RagasJudge, extract_json
from aiact.evals.judged import check_gate, stratified_sample, summarize
from aiact.llm.replay import ReplayLLM, ScriptedLLM

STATEMENTS = json.dumps({"statements": ["Social scoring is prohibited.", "Fines reach 35 million EUR."]})
VERDICTS = json.dumps(
    {
        "statements": [
            {"statement": "Social scoring is prohibited.", "reason": "Article 5(1)(c)", "verdict": 1},
            {"statement": "Fines reach 35 million EUR.", "reason": "not in context", "verdict": 0},
        ]
    }
)


@pytest.mark.parametrize(
    "raw",
    ['{"a": 1}', '```json\n{"a": 1}\n```', 'Here you go: {"a": 1} Hope that helps.'],
)
def test_extract_json(raw):
    assert json.loads(extract_json(raw)) == {"a": 1}


def test_faithfulness_runs_through_our_client_and_passes_the_schema():
    llm = ScriptedLLM([STATEMENTS, VERDICTS])
    judge = RagasJudge(llm)
    res = asyncio.run(
        Faithfulness(llm=judge).ascore(
            user_input="Is social scoring banned?",
            response="Yes, social scoring is prohibited and fines reach 35 million EUR.",
            retrieved_contexts=["Article 5(1)(c) prohibits social scoring."],
        )
    )
    assert res.value == 0.5 and judge.calls == 2
    assert llm.calls[0]["json_schema"] == StatementGeneratorOutput.model_json_schema()
    assert "Is social scoring banned?" in llm.calls[0]["messages"][0]["content"]  # Ragas' own prompt is sent


def test_answer_correctness_factuality_only():
    classification = json.dumps(
        {
            "TP": [{"statement": "Social scoring is prohibited.", "reason": "in reference"}],
            "FP": [{"statement": "Fines reach 35 million EUR.", "reason": "not in reference"}],
            "FN": [],
        }
    )
    reference_statements = json.dumps({"statements": ["Social scoring by anyone is prohibited."]})
    judge = RagasJudge(ScriptedLLM([STATEMENTS, reference_statements, classification]))
    res = asyncio.run(
        AnswerCorrectness(llm=judge, weights=[1.0, 0.0]).ascore(
            user_input="q", response="answer", reference="Social scoring by anyone is prohibited."
        )
    )
    assert res.value == pytest.approx(2 / 3)  # precision 1/2, recall 1/1 -> F1 2/3
    assert judge.calls == 3


def test_invalid_json_gets_one_repair_turn_then_fails_loudly():
    llm = ScriptedLLM(["not json", STATEMENTS])
    judge = RagasJudge(llm)
    out = judge.generate("prompt", StatementGeneratorOutput)
    assert out.statements[0] == "Social scoring is prohibited."
    assert (
        judge.parse_retries == 1
        and "did not match the required JSON schema" in llm.calls[1]["messages"][-1]["content"]
    )
    with pytest.raises(JudgeError):
        RagasJudge(ScriptedLLM(["nope", "still nope"])).generate("prompt", StatementGeneratorOutput)


def test_judge_cassette_replays(tmp_path):
    path = tmp_path / "judge.json"
    RagasJudge(ReplayLLM(path, inner=ScriptedLLM([STATEMENTS]))).generate("p", StatementGeneratorOutput)
    replayed = RagasJudge(ReplayLLM(path)).generate("p", StatementGeneratorOutput)
    assert len(replayed.statements) == 2


def test_stratified_sample_is_proportional_reproducible_and_ordered():
    samples = [{"id": f"h{i:02d}", "category": "high_risk"} for i in range(12)]
    samples += [{"id": f"d{i}", "category": "definitions"} for i in range(6)] + [
        {"id": "l1", "category": "literacy"}
    ]
    a, b = stratified_sample(samples, 8), stratified_sample(samples, 8)
    assert a == b and len(a) == 8
    cats = [s["category"] for s in a]
    assert cats.count("literacy") == 1 and cats.count("high_risk") > cats.count("definitions") >= 1
    assert [s["id"] for s in a] == [s["id"] for s in samples if s in a]
    assert stratified_sample(samples, 100) == samples


def test_summarize_and_gate_handle_judge_failures():
    rows = [
        {"faithfulness": 1.0, "answer_correctness": 0.5},
        {"faithfulness": 0.5, "answer_correctness": 1.0},
        {"faithfulness": math.nan, "answer_correctness": math.nan},
    ]
    m = summarize(rows)
    assert m["faithfulness"] == 0.75 and m["unfaithful_answers"] == 1 and m["judge_failures"] == 1
    assert check_gate(m, {"min": {"faithfulness": 0.7}, "max": {"judge_failures": 1}}) == []
    assert len(check_gate(m | {"faithfulness": math.nan}, {"min": {"faithfulness": 0.7}})) == 1  # NaN fails
