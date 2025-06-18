import argparse
import logging

logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PKB Command Line Interface")

    command_parser = parser.add_subparsers(dest="command")

    ingest_parser = command_parser.add_parser("ingest")
    ingest_parser.add_argument("files", nargs="+", help="Files to ingest")
    ingest_parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop existing database tables before ingesting",
    )

    ask_parser = command_parser.add_parser("ask")
    ask_parser.add_argument("question", help="Question to ask the system")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    match args.command:
        case "ingest":
            from pkb.db import init_cache_db
            from pkb.ingest import ingest_files

            if args.drop:
                logging.info("Dropping existing database tables before ingesting.")
                init_cache_db(drop=args.drop)

            ingest_files(args.files)
        case "ask":
            from pkb.rag import ask_question

            ask_question(args.question)
        case _:
            raise ValueError(f"Unknown command: {args.command}")
