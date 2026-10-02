"""Command-line entry point: ``uv run hmm-markov {data,rain,words,report,all}``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .data import ROOT

RESULTS = ROOT / "results"


def _save(name: str, payload: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    return path


def cmd_data() -> None:
    from .data import fetch_fsdd, load_corpus, load_rain

    wet = load_rain()
    print(f"rain: {len(wet)} five-minute steps ({len(wet) / 288:.0f} days), {wet.mean():.2%} wet")
    print(f"fruits: {len(load_corpus('fruits'))} recordings")
    print(f"fsdd: {len(list(fetch_fsdd().glob('*.wav')))} recordings (pinned v1.0.10)")


def cmd_rain(seed: int) -> None:
    from . import rain

    start = time.perf_counter()
    path = _save("rain", rain.run(seed))
    print(f"rain experiment -> {path.relative_to(ROOT)} ({time.perf_counter() - start:.0f} s)")


def cmd_words(seed: int) -> None:
    from . import words

    start = time.perf_counter()
    path = _save("words", words.run(seed))
    print(f"word experiment -> {path.relative_to(ROOT)} ({time.perf_counter() - start:.0f} s)")


def cmd_report() -> None:
    from . import report

    for path in report.build(RESULTS):
        print(f"wrote {path.relative_to(ROOT)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="hmm-markov", description=__doc__)
    parser.add_argument("command", choices=["data", "rain", "words", "report", "all"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command in ("data", "all"):
        cmd_data()
    if args.command in ("rain", "all"):
        cmd_rain(args.seed)
    if args.command in ("words", "all"):
        cmd_words(args.seed)
    if args.command in ("report", "all"):
        cmd_report()


if __name__ == "__main__":
    main()
