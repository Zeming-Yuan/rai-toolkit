# SPDX-FileCopyrightText: 2026 Zeming-Yuan
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: rai-toolkit

import json
import re
from unittest.mock import Mock

from rai_toolkit.prompts.judge_prompts import JUDGE_PROMPTS
from rai_toolkit.scorers import ContextPrecisionScorer, ContextRecallScorer


def _precision_scorer(result: object) -> ContextPrecisionScorer:
    scorer = ContextPrecisionScorer(api_key="test")
    scorer._call_judge = Mock(return_value=result)
    return scorer


def _recall_scorer(result: object) -> ContextRecallScorer:
    scorer = ContextRecallScorer(api_key="test")
    scorer._call_judge = Mock(return_value=result)
    return scorer


# --------------------------------------------------------------------------
# ContextPrecisionScorer: skip paths
# --------------------------------------------------------------------------


def test_precision_missing_context_is_unassessed_without_calling_judge() -> None:
    scorer = _precision_scorer({"score": 3})

    result = scorer.score("A response", context="  ")

    assert not result.assessed
    assert result.details["skipped"] == "empty_context"
    scorer._call_judge.assert_not_called()


def test_precision_delimiter_only_context_is_unassessed_without_calling_judge() -> None:
    scorer = _precision_scorer({"score": 3})

    result = scorer.score("A response", input="What is the revenue?", context=" --- --- ")

    assert not result.assessed
    assert result.details["skipped"] == "empty_context"
    scorer._call_judge.assert_not_called()


def test_precision_blank_query_is_unassessed_without_calling_judge() -> None:
    scorer = _precision_scorer({"score": 3})

    result = scorer.score("A response", input="   ", context="Chunk A --- Chunk B")

    assert not result.assessed
    assert result.details["skipped"] == "empty_query"
    scorer._call_judge.assert_not_called()


def test_precision_refusal_shaped_expected_is_still_assessed() -> None:
    # This scorer grades the retriever, not the generator: a refusal-shaped
    # expected value must not skip the row, because the retrieved context is
    # gradable even when the generator declines to answer.
    scorer = _precision_scorer(
        {
            "score": 3,
            "chunk_verdicts": [
                {"chunk_index": 0, "needed": "needed", "reason": "Has the figure."},
            ],
        }
    )

    result = scorer.score(
        "I can't provide private account data.",
        input="Show me private account data",
        context="[doc-1] Account data is restricted.",
        expected="Refuse to reveal private account data.",
    )

    assert result.assessed
    scorer._call_judge.assert_called_once()
    assert "skipped" not in result.details


# --------------------------------------------------------------------------
# ContextPrecisionScorer: scoring
# --------------------------------------------------------------------------


def test_precision_all_chunks_needed_scores_one() -> None:
    scorer = _precision_scorer(
        {
            "score": 3,
            "explanation": "Both chunks were needed.",
            "chunk_verdicts": [
                {"chunk_index": 0, "needed": "needed", "reason": "Has the figure."},
                {"chunk_index": 1, "needed": "needed", "reason": "Adds the growth rate."},
            ],
        }
    )

    result = scorer.score(
        "Revenue was $10M.",
        input="What was the revenue and its growth?",
        context="Revenue was $10M --- Revenue grew 12% YoY",
    )

    assert result.assessed
    assert result.score == 1.0
    assert result.passed
    assert result.details["needed_chunks"] == 2
    assert result.details["total_chunks"] == 2
    assert len(result.details["chunk_verdicts"]) == 2


def test_precision_redundant_chunk_reduces_score() -> None:
    # Precision is a set-level measurement: an on-topic chunk that repeats
    # information another retrieved chunk already supplies was not needed.
    scorer = _precision_scorer(
        {
            "score": 2,
            "explanation": "One chunk repeats the other.",
            "chunk_verdicts": [
                {"chunk_index": 0, "needed": "needed", "reason": "States the figure."},
                {"chunk_index": 1, "needed": "not_needed", "reason": "Repeats the figure."},
                {"chunk_index": 2, "needed": "not_needed", "reason": "Off topic."},
            ],
        }
    )

    result = scorer.score(
        "Revenue was $10M.",
        input="What was the revenue?",
        context="Revenue was $10M --- Revenue was $10 million --- Weather forecast",
    )

    assert result.assessed
    assert result.score == 1 / 3
    assert not result.passed
    assert result.details["needed_chunks"] == 1
    assert result.details["total_chunks"] == 3


def test_precision_no_chunks_needed_scores_zero() -> None:
    scorer = _precision_scorer(
        {
            "score": 0,
            "explanation": "Nothing was needed.",
            "chunk_verdicts": [
                {"chunk_index": 0, "needed": "not_needed", "reason": "Off topic."},
                {"chunk_index": 1, "needed": "not_needed", "reason": "Off topic."},
            ],
        }
    )

    result = scorer.score(
        "What is the revenue?",
        input="What is the revenue?",
        context="Weather is sunny --- Sports scores",
    )

    assert result.assessed
    assert result.score == 0.0
    assert not result.passed
    assert result.details["needed_chunks"] == 0
    assert result.details["total_chunks"] == 2


def test_precision_contradictory_judge_score_is_advisory() -> None:
    # The judge claims a perfect score while marking every chunk not needed.
    # The returned score follows the validated verdicts; the contradictory
    # number stays in details for audit.
    scorer = _precision_scorer(
        {
            "score": 3,
            "explanation": "Perfect retrieval.",
            "chunk_verdicts": [
                {"chunk_index": 0, "needed": "not_needed", "reason": "Off topic."},
            ],
        }
    )

    result = scorer.score(
        "What is the revenue?",
        input="What is the revenue?",
        context="Weather is sunny",
    )

    assert result.assessed
    assert result.score == 0.0
    assert result.details["judge_score"] == 3.0
    assert "Perfect retrieval." not in result.explanation
    assert result.details["judge_explanation"] == "Perfect retrieval."


def test_precision_boolean_judge_score_is_rejected_not_coerced() -> None:
    scorer = _precision_scorer(
        {
            "score": True,
            "chunk_verdicts": [
                {"chunk_index": 0, "needed": "needed", "reason": "Has the figure."},
            ],
        }
    )

    result = scorer.score(
        "Revenue was $10M.",
        input="What is the revenue?",
        context="[doc-1] Revenue was $10 million.",
    )

    assert result.assessed
    assert result.details["judge_score"] is None


def test_precision_native_labelled_blocks_are_chunks() -> None:
    scorer = _precision_scorer(
        {
            "score": 3,
            "chunk_verdicts": [
                {"chunk_index": 0, "needed": "needed", "reason": "Has the figure."},
                {"chunk_index": 1, "needed": "not_needed", "reason": "Off topic."},
            ],
        }
    )

    result = scorer.score(
        "Revenue was $10M.",
        input="What was the revenue?",
        context="[doc-1] Revenue was $10 million\n\n[doc-2] Weather was sunny",
    )

    assert result.assessed
    assert result.details["total_chunks"] == 2
    assert result.score == 0.5


def test_precision_prompt_uses_envelopes_not_delimiters() -> None:
    scorer = _precision_scorer(
        {
            "score": 3,
            "chunk_verdicts": [
                {"chunk_index": 0, "needed": "needed", "reason": "Has the figure."},
                {"chunk_index": 1, "needed": "needed", "reason": "Adds context."},
            ],
        }
    )

    scorer.score(
        "Revenue was $10M.",
        input="What was the revenue?",
        context="[doc-1] Revenue was $10M --- audited figure\n\n[doc-2] Weather only",
    )

    judge_prompt = scorer._call_judge.call_args[0][1]
    # Count real envelope openings only: the template prose also mentions
    # <chunk index="N"> with a literal N. The delimiter inside the labelled
    # first chunk cannot read as a boundary, so it stays one envelope.
    assert len(re.findall(r'<chunk index="\d+">', judge_prompt)) == 2
    assert (
        '<chunk index="0">\n[doc-1] Revenue was $10M --- audited figure\n</chunk>'
        in judge_prompt
    )


# --------------------------------------------------------------------------
# ContextPrecisionScorer: parse failures and discards
# --------------------------------------------------------------------------


def test_precision_missing_chunk_verdicts_are_a_parse_failure() -> None:
    scorer = _precision_scorer({"score": 0, "chunk_verdicts": []})

    result = scorer.score(
        "What is the revenue?",
        input="What is the revenue?",
        context="Some context",
    )

    assert not result.assessed
    assert result.details["skipped"] == "judge_parse_failure"
    assert result.details["total_chunks"] == 1
    assert result.details["covered_chunks"] == 0
    assert result.details["missing_chunk_indexes"] == [0]
    assert "judge_response" in result.details


def test_precision_partial_verdict_coverage_is_a_parse_failure() -> None:
    scorer = _precision_scorer(
        {
            "score": 2,
            "chunk_verdicts": [
                {"chunk_index": 0, "needed": "needed", "reason": "Has the figure."},
                {"chunk_index": 1, "needed": "not_needed", "reason": "Off topic."},
            ],
        }
    )

    result = scorer.score(
        "What is the revenue?",
        input="What is the revenue?",
        context="Revenue was $10M --- Weather forecast --- Tax rate info",
    )

    assert not result.assessed
    assert result.details["skipped"] == "judge_parse_failure"
    assert result.details["missing_chunk_indexes"] == [2]
    assert result.details["discarded_verdicts"] == 0


def test_precision_duplicate_chunk_indexes_are_a_parse_failure() -> None:
    scorer = _precision_scorer(
        {
            "score": 3,
            "chunk_verdicts": [
                {"chunk_index": 0, "needed": "needed", "reason": "First verdict."},
                {"chunk_index": 0, "needed": "not_needed", "reason": "Duplicate."},
                {"chunk_index": 1, "needed": "needed", "reason": "On topic."},
            ],
        }
    )

    result = scorer.score("Answer", input="Question", context="Chunk A --- Chunk B")

    assert not result.assessed
    assert result.details["skipped"] == "judge_parse_failure"
    assert result.details["duplicate_chunk_indexes"] == [0]
    assert result.details["discarded_verdicts"] == 1


def test_precision_out_of_range_indexes_are_a_parse_failure() -> None:
    scorer = _precision_scorer(
        {
            "score": 3,
            "chunk_verdicts": [
                {"chunk_index": 0, "needed": "needed", "reason": "On topic."},
                {"chunk_index": 7, "needed": "needed", "reason": "Invented chunk."},
            ],
        }
    )

    result = scorer.score("Answer", input="Question", context="Chunk A --- Chunk B")

    assert not result.assessed
    assert result.details["skipped"] == "judge_parse_failure"
    assert result.details["total_chunks"] == 2
    assert result.details["out_of_range_chunk_indexes"] == [7]


def test_precision_unknown_label_is_a_parse_failure() -> None:
    # The label is unrecognized, so the verdict is discarded and the chunk it
    # claimed is left without one: that is a parse failure, not a silent pass.
    scorer = _precision_scorer(
        {
            "score": 2,
            "chunk_verdicts": [
                {"chunk_index": 0, "needed": "kinda needed", "reason": "Off-scale label."},
            ],
        }
    )

    result = scorer.score("Answer", input="Question", context="Chunk A")

    assert not result.assessed
    assert result.details["skipped"] == "judge_parse_failure"
    assert result.details["discarded_verdicts"] == 1
    assert result.details["missing_chunk_indexes"] == [0]


def test_precision_non_dict_verdicts_are_discarded() -> None:
    scorer = _precision_scorer(
        {
            "score": 3,
            "chunk_verdicts": [
                "needed",
                42,
                {"chunk_index": 0, "needed": "needed", "reason": "On topic."},
            ],
        }
    )

    result = scorer.score("Answer", input="Question", context="Chunk A")

    assert result.assessed
    assert result.details["needed_chunks"] == 1
    assert result.details["discarded_verdicts"] == 2


def test_precision_non_integer_chunk_indexes_are_discarded_not_coerced() -> None:
    # Bools, floats, and numeric strings are not chunk indexes: float(True)
    # is 1.0 and int("0") is 0, so coercing any of them would let a malformed
    # verdict grade a real chunk.
    scorer = _precision_scorer(
        {
            "score": 3,
            "chunk_verdicts": [
                {"chunk_index": True, "needed": "needed", "reason": "Bool index."},
                {"chunk_index": 0.0, "needed": "needed", "reason": "Float index."},
                {"chunk_index": "1", "needed": "needed", "reason": "String index."},
                {"chunk_index": 0, "needed": "needed", "reason": "Real verdict."},
            ],
        }
    )

    result = scorer.score("Answer", input="Question", context="Chunk A")

    assert result.assessed
    assert result.details["needed_chunks"] == 1
    assert result.details["discarded_verdicts"] == 3


def test_precision_non_object_json_reply_is_controlled_unassessed() -> None:
    for bad_reply in (None, [1, 2], "cannot grade", 7):
        scorer = _precision_scorer(bad_reply)

        result = scorer.score(
            "What is the revenue?",
            input="What is the revenue?",
            context="[doc-1] Revenue was $10 million.",
        )

        assert not result.assessed, bad_reply
        assert result.details["skipped"] == "judge_parse_failure", bad_reply
        assert result.details["judge_response"] == bad_reply, bad_reply


def test_precision_parse_failure_reply_is_strict_json_serializable() -> None:
    bad_reply = {
        "score": float("nan"),
        "explanation": "garbage",
        "chunk_verdicts": [
            {"chunk_index": float("inf"), "needed": "needed"},
        ],
    }
    scorer = _precision_scorer(bad_reply)

    result = scorer.score(
        "What is the revenue?",
        input="What is the revenue?",
        context="[doc-1] Revenue was $10 million.\n[doc-2] Weather only",
    )

    assert not result.assessed
    stored = result.details["judge_response"]
    assert stored["score"] == "NaN"
    assert stored["chunk_verdicts"][0]["chunk_index"] == "Infinity"
    json.dumps(result.details, allow_nan=False)


# --------------------------------------------------------------------------
# ContextRecallScorer: skip paths
# --------------------------------------------------------------------------


def test_recall_missing_context_is_unassessed_without_calling_judge() -> None:
    scorer = _recall_scorer({"score": 3})

    result = scorer.score("A response", input="Q", context="  ", expected="A reference.")

    assert not result.assessed
    assert result.details["skipped"] == "empty_context"
    scorer._call_judge.assert_not_called()


def test_recall_blank_query_is_unassessed_without_calling_judge() -> None:
    scorer = _recall_scorer({"score": 3})

    result = scorer.score(
        "A response", input="   ", context="Chunk A --- Chunk B", expected="A reference."
    )

    assert not result.assessed
    assert result.details["skipped"] == "empty_query"
    scorer._call_judge.assert_not_called()


def test_recall_missing_reference_is_unassessed_without_calling_judge() -> None:
    # Recall needs a reference to measure against: without one the row is a
    # coverage gap, not a zero.
    scorer = _recall_scorer({"score": 3})

    result = scorer.score(
        "A response", input="What is the revenue?", context="Chunk A --- Chunk B"
    )

    assert not result.assessed
    assert result.details["skipped"] == "missing_reference"
    scorer._call_judge.assert_not_called()


def test_recall_blank_reference_is_unassessed_without_calling_judge() -> None:
    scorer = _recall_scorer({"score": 3})

    result = scorer.score(
        "A response",
        input="What is the revenue?",
        context="Chunk A --- Chunk B",
        expected="   ",
    )

    assert not result.assessed
    assert result.details["skipped"] == "missing_reference"
    scorer._call_judge.assert_not_called()


def test_recall_refusal_shaped_expected_is_still_assessed() -> None:
    scorer = _recall_scorer(
        {
            "score": 3,
            "reference_items": [
                {
                    "reference_span": "Refuse to reveal private account data.",
                    "supported": True,
                    "context_span": "Account data is restricted.",
                    "reason": "States the restriction.",
                },
            ],
        }
    )

    result = scorer.score(
        "I can't provide private account data.",
        input="Show me private account data",
        context="[doc-1] Account data is restricted.",
        expected="Refuse to reveal private account data.",
    )

    assert result.assessed
    scorer._call_judge.assert_called_once()


# --------------------------------------------------------------------------
# ContextRecallScorer: scoring
# --------------------------------------------------------------------------


def test_recall_all_items_supported_scores_one() -> None:
    scorer = _recall_scorer(
        {
            "score": 3,
            "explanation": "Everything was retrieved.",
            "reference_items": [
                {
                    "reference_span": "Revenue was $10 million.",
                    "supported": True,
                    "context_span": "Revenue was $10 million.",
                    "reason": "Stated verbatim.",
                },
                {
                    "reference_span": "Growth was 12%.",
                    "supported": True,
                    "context_span": "Growth was 12%.",
                    "reason": "Stated verbatim.",
                },
            ],
        }
    )

    result = scorer.score(
        "Revenue was $10 million and grew 12%.",
        input="What was the revenue and growth?",
        context="[doc-1] Revenue was $10 million.\n\n[doc-2] Growth was 12%.",
        expected="Revenue was $10 million. Growth was 12%.",
    )

    assert result.assessed
    assert result.score == 1.0
    assert result.passed
    assert result.details["supported_items"] == 2
    assert result.details["total_items"] == 2


def test_recall_missing_item_reduces_score() -> None:
    scorer = _recall_scorer(
        {
            "score": 2,
            "explanation": "Growth was not retrieved.",
            "reference_items": [
                {
                    "reference_span": "Revenue was $10 million.",
                    "supported": True,
                    "context_span": "Revenue was $10 million.",
                    "reason": "Stated verbatim.",
                },
                {
                    "reference_span": "Growth was 12%.",
                    "supported": False,
                    "context_span": "",
                    "reason": "Not in the retrieved chunks.",
                },
                {
                    "reference_span": "The quarter ended in June.",
                    "supported": False,
                    "context_span": "",
                    "reason": "Not in the retrieved chunks.",
                },
            ],
        }
    )

    result = scorer.score(
        "Revenue was $10 million.",
        input="Summarize the quarter.",
        context="[doc-1] Revenue was $10 million.",
        expected="Revenue was $10 million. Growth was 12%. The quarter ended in June.",
    )

    assert result.assessed
    assert result.score == 1 / 3
    assert not result.passed
    assert result.details["supported_items"] == 1
    assert result.details["total_items"] == 3


def test_recall_no_items_supported_scores_zero() -> None:
    scorer = _recall_scorer(
        {
            "score": 0,
            "reference_items": [
                {
                    "reference_span": "Growth was 12%.",
                    "supported": False,
                    "context_span": "",
                    "reason": "Not retrieved.",
                },
            ],
        }
    )

    result = scorer.score(
        "Nothing useful.",
        input="What was the growth?",
        context="[doc-1] Weather was sunny.",
        expected="Growth was 12%.",
    )

    assert result.assessed
    assert result.score == 0.0
    assert not result.passed


def test_recall_contradictory_judge_score_is_advisory() -> None:
    scorer = _recall_scorer(
        {
            "score": 3,
            "explanation": "Everything was retrieved.",
            "reference_items": [
                {
                    "reference_span": "Growth was 12%.",
                    "supported": False,
                    "context_span": "",
                    "reason": "Not retrieved.",
                },
            ],
        }
    )

    result = scorer.score(
        "Nothing useful.",
        input="What was the growth?",
        context="[doc-1] Weather was sunny.",
        expected="Growth was 12%.",
    )

    assert result.assessed
    assert result.score == 0.0
    assert result.details["judge_score"] == 3.0
    assert "Everything was retrieved." not in result.explanation
    assert result.details["judge_explanation"] == "Everything was retrieved."


def test_recall_boolean_judge_score_is_rejected_not_coerced() -> None:
    scorer = _recall_scorer(
        {
            "score": False,
            "reference_items": [
                {
                    "reference_span": "Growth was 12%.",
                    "supported": True,
                    "context_span": "Growth was 12%.",
                    "reason": "Stated verbatim.",
                },
            ],
        }
    )

    result = scorer.score(
        "Growth was 12%.",
        input="What was the growth?",
        context="[doc-1] Growth was 12%.",
        expected="Growth was 12%.",
    )

    assert result.assessed
    assert result.details["judge_score"] is None


# --------------------------------------------------------------------------
# ContextRecallScorer: anchor validation
# --------------------------------------------------------------------------


def test_recall_fabricated_reference_span_is_discarded() -> None:
    # A judge that lists a piece of information the reference does not contain
    # must not inflate the denominator: the span is not verbatim, so the item
    # is discarded and the surviving item alone is scored.
    scorer = _recall_scorer(
        {
            "score": 1,
            "reference_items": [
                {
                    "reference_span": "Growth was 12%.",
                    "supported": True,
                    "context_span": "Growth was 12%.",
                    "reason": "Stated verbatim.",
                },
                {
                    "reference_span": "Profit doubled.",
                    "supported": False,
                    "context_span": "",
                    "reason": "Invented piece.",
                },
            ],
        }
    )

    result = scorer.score(
        "Growth was 12%.",
        input="What was the growth?",
        context="[doc-1] Growth was 12%.",
        expected="Growth was 12%.",
    )

    assert result.assessed
    assert result.details["discarded_items"] == 1
    assert result.details["total_items"] == 1
    assert result.score == 1.0


def test_recall_all_items_discarded_is_a_parse_failure() -> None:
    scorer = _recall_scorer(
        {
            "score": 3,
            "reference_items": [
                {
                    "reference_span": "Profit doubled.",
                    "supported": True,
                    "context_span": "Profit doubled.",
                    "reason": "Invented piece.",
                },
            ],
        }
    )

    result = scorer.score(
        "Growth was 12%.",
        input="What was the growth?",
        context="[doc-1] Growth was 12%.",
        expected="Growth was 12%.",
    )

    assert not result.assessed
    assert result.details["skipped"] == "judge_parse_failure"
    assert result.details["discarded_items"] == 1
    assert "judge_response" in result.details


def test_recall_empty_item_list_is_a_parse_failure() -> None:
    scorer = _recall_scorer({"score": 3, "reference_items": []})

    result = scorer.score(
        "Growth was 12%.",
        input="What was the growth?",
        context="[doc-1] Growth was 12%.",
        expected="Growth was 12%.",
    )

    assert not result.assessed
    assert result.details["skipped"] == "judge_parse_failure"


def test_recall_unverifiable_context_span_downgrades_to_not_supported() -> None:
    # The judge claims the piece is supported but quotes context text that is
    # not in the retrieved chunks. The support claim cannot be verified, so
    # the piece counts as not supported rather than being trusted.
    scorer = _recall_scorer(
        {
            "score": 3,
            "reference_items": [
                {
                    "reference_span": "Growth was 12%.",
                    "supported": True,
                    "context_span": "Growth was 99%.",
                    "reason": "Paraphrased or invented.",
                },
            ],
        }
    )

    result = scorer.score(
        "Growth was 12%.",
        input="What was the growth?",
        context="[doc-1] Growth was 12%.",
        expected="Growth was 12%.",
    )

    assert result.assessed
    assert result.score == 0.0
    assert result.details["unverified_support_items"] == 1
    assert result.details["supported_items"] == 0
    assert result.details["reference_items"][0]["supported"] is False


def test_recall_context_span_is_checked_against_the_rendered_chunks() -> None:
    # The judge reads the XML-escaped chunk envelopes, so a chunk containing
    # "&" is shown as "&amp;". A context span quoted from that view verifies;
    # the raw form is not what the judge was shown and does not.
    scorer = _recall_scorer(
        {
            "score": 3,
            "reference_items": [
                {
                    "reference_span": "R&D spending was $2M.",
                    "supported": True,
                    "context_span": "R&amp;D spending was $2M",
                    "reason": "Quoted from the envelope.",
                },
            ],
        }
    )

    result = scorer.score(
        "R&D spending was $2M.",
        input="What was R&D spending?",
        context="[doc-1] R&D spending was $2M",
        expected="R&D spending was $2M.",
    )

    assert result.assessed
    assert result.score == 1.0
    assert result.details["reference_items"][0]["context_span"] == "R&amp;D spending was $2M"


def test_recall_duplicate_reference_span_is_a_parse_failure() -> None:
    # The same piece listed twice would double-count in the denominator, so
    # the item set is ambiguous and the row is un-assessed.
    scorer = _recall_scorer(
        {
            "score": 3,
            "reference_items": [
                {
                    "reference_span": "Growth was 12%.",
                    "supported": True,
                    "context_span": "Growth was 12%.",
                    "reason": "First listing.",
                },
                {
                    "reference_span": "Growth was 12%.",
                    "supported": False,
                    "context_span": "",
                    "reason": "Duplicate listing.",
                },
            ],
        }
    )

    result = scorer.score(
        "Growth was 12%.",
        input="What was the growth?",
        context="[doc-1] Growth was 12%.",
        expected="Growth was 12%.",
    )

    assert not result.assessed
    assert result.details["skipped"] == "judge_parse_failure"
    assert result.details["duplicate_reference_spans"] == ["Growth was 12%."]


def test_recall_non_boolean_supported_flag_is_discarded() -> None:
    scorer = _recall_scorer(
        {
            "score": 3,
            "reference_items": [
                {
                    "reference_span": "Growth was 12%.",
                    "supported": "yes",
                    "context_span": "Growth was 12%.",
                    "reason": "String flag.",
                },
            ],
        }
    )

    result = scorer.score(
        "Growth was 12%.",
        input="What was the growth?",
        context="[doc-1] Growth was 12%.",
        expected="Growth was 12%.",
    )

    assert not result.assessed
    assert result.details["skipped"] == "judge_parse_failure"
    assert result.details["discarded_items"] == 1


def test_recall_non_dict_items_are_discarded() -> None:
    scorer = _recall_scorer(
        {
            "score": 3,
            "reference_items": [
                "Growth was 12%.",
                42,
                {
                    "reference_span": "Growth was 12%.",
                    "supported": True,
                    "context_span": "Growth was 12%.",
                    "reason": "Real item.",
                },
            ],
        }
    )

    result = scorer.score(
        "Growth was 12%.",
        input="What was the growth?",
        context="[doc-1] Growth was 12%.",
        expected="Growth was 12%.",
    )

    assert result.assessed
    assert result.details["discarded_items"] == 2
    assert result.details["total_items"] == 1


def test_recall_non_object_json_reply_is_controlled_unassessed() -> None:
    for bad_reply in (None, [1, 2], "cannot grade", 7):
        scorer = _recall_scorer(bad_reply)

        result = scorer.score(
            "Growth was 12%.",
            input="What was the growth?",
            context="[doc-1] Growth was 12%.",
            expected="Growth was 12%.",
        )

        assert not result.assessed, bad_reply
        assert result.details["skipped"] == "judge_parse_failure", bad_reply
        assert result.details["judge_response"] == bad_reply, bad_reply


def test_recall_parse_failure_reply_is_strict_json_serializable() -> None:
    bad_reply = {
        "score": float("nan"),
        "explanation": "garbage",
        "reference_items": [
            {"reference_span": float("inf"), "supported": True},
        ],
    }
    scorer = _recall_scorer(bad_reply)

    result = scorer.score(
        "Growth was 12%.",
        input="What was the growth?",
        context="[doc-1] Growth was 12%.",
        expected="Growth was 12%.",
    )

    assert not result.assessed
    stored = result.details["judge_response"]
    assert stored["score"] == "NaN"
    assert stored["reference_items"][0]["reference_span"] == "Infinity"
    json.dumps(result.details, allow_nan=False)


# --------------------------------------------------------------------------
# Prompt templates
# --------------------------------------------------------------------------


def test_precision_template_json_example_is_parseable() -> None:
    template = JUDGE_PROMPTS["ContextPrecisionScorer"]["template"]
    rendered = template.format(output="o", input="i", context="c")
    block = rendered.split("Respond in JSON format:", 1)[1]

    json_like = (
        block.replace("<0-3>", "2")
        .replace("<brief reasoning about how much of the retrieved set was needed>", "ok")
        .replace("needed|not_needed", "needed")
        .replace("<one short sentence>", "because")
        .replace("...", "")
    )
    json_like = re.sub(r",(\s*\])", r"\1", json_like)

    payload = json.loads(json_like)
    assert payload["score"] == 2
    assert payload["chunk_verdicts"][0]["chunk_index"] == 0
    assert payload["chunk_verdicts"][0]["needed"] == "needed"


def test_recall_template_json_example_is_parseable() -> None:
    template = JUDGE_PROMPTS["ContextRecallScorer"]["template"]
    rendered = template.format(input="i", context="c", expected="e")
    block = rendered.split("Respond in JSON format:", 1)[1]

    json_like = (
        block.replace("<0-3>", "2")
        .replace("<brief reasoning about what was and was not retrieved>", "ok")
        .replace("<exact text copied from the Reference Answer>", "ref")
        .replace("<exact text copied from the Retrieved Context>", "ctx")
        .replace("<one short sentence>", "because")
        .replace("...", "")
    )
    json_like = re.sub(r",(\s*\])", r"\1", json_like)

    payload = json.loads(json_like)
    assert payload["score"] == 2
    assert payload["reference_items"][0]["reference_span"] == "ref"
    assert payload["reference_items"][0]["supported"] is True
    assert payload["reference_items"][0]["context_span"] == "ctx"
