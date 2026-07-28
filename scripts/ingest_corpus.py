"""CLI script to batch ingest all PDF documents in a directory.

Usage:
    uv run python scripts/ingest_corpus.py [path_to_directory] [--strategy structural]
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragchat.ingestion.graph import ingestion_graph
from ragchat.logging_conf import configure_logging
import structlog

configure_logging()
logger = structlog.get_logger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ingest PDFs into RAG system.")
    parser.add_argument(
        "dir_path",
        type=str,
        nargs="?",
        default="doc",
        help="Path to directory containing PDFs (default: doc)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="structural",
        choices=["fixed_size", "recursive", "structural", "semantic"],
        help="Chunking strategy to use (default: structural)",
    )
    args = parser.parse_args()

    dir_path = Path(args.dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        logger.error("directory_not_found", path=str(dir_path))
        sys.exit(1)

    pdf_files = list(dir_path.glob("*.pdf"))
    if not pdf_files:
        logger.warn("no_pdf_files_found", path=str(dir_path))
        return

    logger.info("found_pdfs_for_ingestion", count=len(pdf_files), strategy=args.strategy)

    for pdf in pdf_files:
        print(f"\nIngesting {pdf.name} using {args.strategy} chunking strategy...")
        try:
            # Execute Ingestion StateGraph
            result = await ingestion_graph.ainvoke({
                "pdf_path": str(pdf),
                "chunking_strategy": args.strategy,
            })
            if "error" in result and result["error"]:
                print(f"FAILED: {pdf.name}. Error: {result['error']}")
            else:
                print(f"SUCCESS: {pdf.name} fully ingested and indexed.")
        except Exception as exc:
            logger.error("ingestion_command_failed", pdf=pdf.name, error=str(exc))
            print(f"FAILED: {pdf.name} due to exception: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
