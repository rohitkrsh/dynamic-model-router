from __future__ import annotations

import argparse

from .router import Router


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Route a request to a suitable model")
    parser.add_argument("prompt", help="The prompt or task to classify")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    decision = Router().route(args.prompt)
    print(f"provider={decision.selected_provider}")
    print(f"category={decision.category}")
    print(f"confidence={decision.confidence:.2f}")
    print(f"reason={decision.reason}")


if __name__ == "__main__":
    main()
