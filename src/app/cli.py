from __future__ import annotations

import argparse
from pathlib import Path

from app.graph import ShoppingAssistant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Student scaffold CLI.")
    parser.add_argument("--question", help="Run one question through the graph.")
    parser.add_argument("--test-file", default="data/test.json")
    parser.add_argument("--trace-file", default=None)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    assistant = ShoppingAssistant()

    if args.batch:
        output_dir = assistant.settings.traces_dir
        summary = assistant.run_batch(
            test_file=Path(args.test_file),
            output_dir=output_dir,
            rebuild_index=args.rebuild_index,
        )
        print(f"\nBatch complete: {summary['total']} questions")
        print(f"Traces saved to: {output_dir}")
        print(f"Summary: {output_dir / 'summary.json'}")

    elif args.question:
        trace_file = Path(args.trace_file) if args.trace_file else None
        payload = assistant.ask(
            question=args.question,
            trace_file=trace_file,
            rebuild_index=args.rebuild_index,
        )
        print("\n" + "=" * 60)
        print(payload["final_answer"])
        print("=" * 60)
        if trace_file:
            print(f"Trace saved to: {trace_file}")

    else:
        print("Usage: --question '...' or --batch")
        print("       Add --rebuild-index to force re-embed the policy.")


if __name__ == "__main__":
    main()
