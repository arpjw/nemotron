"""Development-only deterministic systems; these are not model-only predictions."""
from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

VERSION = "1.0.1"
AMOUNT = re.compile(r"(?<![\w.,])(?:\((?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\.[0-9]{2}\)|-?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\.[0-9]{2})(?![\w.,])")
YEAR = re.compile(r"(?<![\w.,])(?:19|20)[0-9]{2}(?![\w.])")
ENTITY = re.compile(r"Fictional_[0-9a-f]{20}")
SCALES = {"dollars": Decimal(1), "thousands": Decimal(1000), "millions": Decimal(1000000)}


class ParseError(ValueError):
    pass


def one(pattern: str, text: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ParseError("missing or ambiguous public header")
    return matches[0]


def public_problem(prompt: str) -> dict[str, Any]:
    """Read literal public text only. No generator, oracle, row metadata or labels."""
    if not isinstance(prompt, str) or len(prompt) > 100_000:
        raise ParseError("invalid prompt")
    entity = one(r"^Entity: (Fictional_[0-9a-f]{20})$", prompt)
    scale = one(r"^All snippet amounts are in USD (dollars|thousands|millions)\.$", prompt)
    question = one(r"^Question: (.+)$", prompt)
    escaped = re.escape(entity)
    rules = [
        ("extract", rf"What was {escaped}'s revenue in ([0-9]{{4}}), in USD dollars\?"),
        ("difference", rf"What was {escaped}'s revenue in ([0-9]{{4}}) minus its revenue in ([0-9]{{4}}), in USD dollars\?"),
        ("growth", rf"What was the percentage change in {escaped}'s revenue from ([0-9]{{4}}) to ([0-9]{{4}})\?"),
    ]
    selected = [(operation, re.fullmatch(pattern, question)) for operation, pattern in rules]
    selected = [(operation, match) for operation, match in selected if match is not None]
    if len(selected) != 1:
        raise ParseError("unsupported or ambiguous public question")
    operation, match = selected[0]
    years = list(match.groups())
    if operation == "difference":
        years.reverse()  # old, new operands
    if len(years) != len(set(years)):
        raise ParseError("repeated question periods")
    lines = re.findall(r"^(L[1-9][0-9]*): (.*)$", prompt, flags=re.MULTILINE)
    if not lines or len({line_id for line_id, _ in lines}) != len(lines):
        raise ParseError("missing or duplicate line IDs")
    # Reject malformed source-like lines rather than silently ignoring them.
    if len(re.findall(r"^L[0-9]+:", prompt, flags=re.MULTILINE)) != len(lines):
        raise ParseError("malformed source line")
    facts = []
    for line_id, text in lines:
        if re.search(r"\brevenue\b", text, flags=re.IGNORECASE) is None:
            continue
        entities = ENTITY.findall(text)
        if len(entities) != 1:
            raise ParseError("ambiguous revenue-line entity")
        if entities[0] != entity:
            continue
        # An exact revenue metric is required: deferred revenue/revenue growth
        # and negated or quoted instructions are not interchangeable facts.
        number = r"(?:\((?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\.[0-9]{2}\)|-?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\.[0-9]{2})"
        year = r"(?:19|20)[0-9]{2}"
        literal = re.escape(entity)
        supported = (
            rf"{literal} revenue {year} {number}",
            rf"\* revenue: {number} \({year}, {literal}\)",
            rf"In {year}, {number} was recorded as {literal}'s revenue\.",
            rf"\[{year}\] \[{literal}\] \[{number}\] \[revenue\]",
        )
        if not any(re.fullmatch(pattern, text) for pattern in supported):
            raise ParseError("unsupported or ambiguous revenue metric syntax")
        stripped = text.replace(entity, "")
        amounts = AMOUNT.findall(stripped)
        periods = YEAR.findall(stripped)
        if len(amounts) != 1 or len(periods) != 1:
            raise ParseError("ambiguous revenue-line amount or period")
        amount = amounts[0]
        negative = amount.startswith("(")
        value = Decimal(amount.strip("()").replace(",", ""))
        if negative:
            value = -value
        facts.append({"line_id": line_id, "period": periods[0], "value": value * SCALES[scale], "raw_amount": amount})
    return {"operation": operation, "periods": years, "entity": entity, "scale": scale, "facts": facts}


def calculate(prompt: str, model_prediction: str | None = None) -> tuple[str, dict[str, Any]]:
    """Return JSON or empty failed prediction, plus an auditable calculation record."""
    from .scoring import parse_response
    log: dict[str, Any] = {"mode": "parser_only" if model_prediction is None else "model_evidence_calculator", "status": "parser_error"}
    try:
        problem = public_problem(prompt)
        log.update({key: value for key, value in problem.items() if key != "facts"})
        operands = []
        reason = None
        for period in problem["periods"]:
            matching = [fact for fact in problem["facts"] if fact["period"] == period]
            if not matching:
                reason = "missing_required_fact"
                break
            if len(matching) > 1:
                if len({fact["value"] for fact in matching}) == 1:
                    raise ParseError("duplicate identical facts have ambiguous citation policy")
                reason = "conflicting_required_facts"
                break
            operands.append(matching[0])
        if reason is None and problem["operation"] == "growth" and operands[0]["value"] <= 0:
            reason = "nonpositive_growth_denominator"
        evidence = sorted((item["line_id"] for item in operands), key=lambda x: int(x[1:])) if reason is None else []
        status = "ok" if reason is None else "unanswerable"
        if model_prediction is not None:
            model = parse_response(model_prediction)
            if model is None:
                raise ParseError("invalid model JSON")
            if model["status"] != status or set(model["evidence"]) != set(evidence):
                raise ParseError("model status or citations fail public-source validation")
            log["model_value"] = model["value"]
            log["model_evidence"] = model["evidence"]
        log["operands"] = [{**item, "value": str(item["value"])} for item in operands]
        if reason is not None:
            result = {"status": "unanswerable", "value": None, "unit": None, "evidence": []}
        else:
            values = [item["value"] for item in operands]
            value = values[0] if problem["operation"] == "extract" else values[1] - values[0]
            if problem["operation"] == "growth":
                value = value / values[0] * 100
            value = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            if value == 0:
                value = abs(value)
            result = {"status": "ok", "value": format(value, ".2f"), "unit": "percent" if problem["operation"] == "growth" else "USD", "evidence": evidence}
        log.update(status="completed", abstention_reason=reason, result=result,
                   value_overridden=model_prediction is not None and log.get("model_value") != result["value"])
        return json.dumps(result, sort_keys=True, separators=(",", ":")), log
    except (ParseError, ArithmeticError) as exc:
        log["error"] = str(exc)
        return "", log


def main() -> None:
    from .model import load_frozen_records, sha256_file
    from .scoring import evaluate_predictions, validate_comparison_provenance
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.data.name != "dev.jsonl":
        parser.error("this exploratory evaluator accepts only frozen dev.jsonl")
    rows = load_frozen_records(args.data)
    source = validate_comparison_provenance(args.predictions, args.data, [row["id"] for row in rows])
    raw = {}
    for line in args.predictions.read_text().splitlines():
        item = json.loads(line)
        if set(item) != {"id", "prediction"} or not isinstance(item["id"], str) or not isinstance(item["prediction"], str) or item["id"] in raw:
            raise ValueError("invalid or duplicate raw prediction row")
        raw[item["id"]] = item["prediction"]
    if set(raw) != {row["id"] for row in rows}:
        raise ValueError("raw predictions must cover the full development split")
    args.output.mkdir(parents=True, exist_ok=False)
    scores = {}
    for mode in ("parser_only", "model_evidence_calculator"):
        predictions = {}
        with (args.output / f"{mode}.jsonl").open("x") as output, (args.output / f"{mode}_audit.jsonl").open("x") as audit:
            for row in rows:
                # The calculator receives ONLY literal public prompt and raw model text.
                prediction, details = calculate(row["prompt"], None if mode == "parser_only" else raw[row["id"]])
                predictions[row["id"]] = prediction
                output.write(json.dumps({"id": row["id"], "prediction": prediction}) + "\n")
                audit.write(json.dumps({"id": row["id"], **details}) + "\n")
        # References are used for scoring here; calculator inputs exclude labels.
        scores[mode] = evaluate_predictions(rows, predictions)
    report = {"status": "completed", "exploratory_dev_only": True, "model_only": False, "calculator_version": VERSION,
              "limitation": "Finite synthetic-question grammar and public numeric-line parser; not general financial document understanding. Calculator-only baseline must accompany assisted results.",
              "data_sha256": sha256_file(args.data), "raw_predictions_sha256": sha256_file(args.predictions),
              "source_inference": source, "calculator_sha256": sha256_file(Path(__file__)),
              "artifact_sha256": {p.name: sha256_file(p) for p in sorted(args.output.glob("*.jsonl"))}, "scores": scores}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({mode: score["rates"] for mode, score in scores.items()}, indent=2))


if __name__ == "__main__":
    main()
