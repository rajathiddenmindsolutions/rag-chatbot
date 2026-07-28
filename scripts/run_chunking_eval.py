"""CLI script to run chunking evaluation across strategies and print comparison report.

Usage:
    uv run python scripts/run_chunking_eval.py
"""

import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docling.document_converter import DocumentConverter
from ragchat.chunking.evaluator import ChunkingEvaluator
from ragchat.chunking.fixed_size import FixedSizeChunker
from ragchat.chunking.recursive import RecursiveMarkdownChunker
from ragchat.chunking.structural import StructuralChunker
from ragchat.chunking.semantic import SemanticChunker


def main() -> None:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    print("Initializing Document Converter & Evaluator...")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    evaluator = ChunkingEvaluator()

    # Find first PDF in doc directory, prefer MOE
    doc_dir = Path(__file__).parent.parent / "doc"
    pdf_files = list(doc_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in doc/ directory.")
        sys.exit(1)
        
    # Prefer MOE Pdf For RAG.pdf since it is smaller and faster to parse
    pdf_path = pdf_files[0]
    for f in pdf_files:
        if "moe" in f.name.lower():
            pdf_path = f
            break
            
    print(f"Evaluating chunking strategies using document: {pdf_path.name}")

    # Parse document with Docling
    print("Parsing document (this may take a minute on CPU)...")
    result = converter.convert(pdf_path)
    docling_doc = result.document
    markdown_text = result.document.export_to_markdown()

    # Define test queries depending on which PDF is processed
    # We mix AGI/ASI and MoE queries so either file yields matches
    queries = [
        {
            "query": "What is the role of Mixture of Experts (MoE) in scaling models?",
            "keywords": ["expert", "moe", "mixture", "routing", "gate"]
        },
        {
            "query": "What is the transition timeline from AGI to ASI?",
            "keywords": ["agi", "asi", "superintelligence", "timeline", "intelligence"]
        },
        {
            "query": "How does the gating network or router select experts?",
            "keywords": ["gate", "gating", "router", "routing", "select"]
        },
        {
            "query": "What are safety and alignment risks for artificial superintelligence?",
            "keywords": ["safety", "alignment", "risk", "existential", "concern"]
        }
    ]

    # Instantiate chunkers
    chunkers = {
        "Fixed-Size Window": FixedSizeChunker(chunk_size=512, chunk_overlap=50),
        "Recursive Markdown": RecursiveMarkdownChunker(chunk_size=1000, chunk_overlap=100),
        "Docling Structural": StructuralChunker(target_chunk_size=1200),
        "Semantic Similarity": SemanticChunker(similarity_threshold_percentile=40.0)
    }

    results = []
    for name, chunker in chunkers.items():
        print(f"Evaluating {name} strategy...")
        res = evaluator.evaluate_strategy(name, chunker, docling_doc, markdown_text, queries)
        results.append(res)

    # Print results report
    print("\n" + "="*80)
    print(" CHUNKING STRATEGY EVALUATION REPORT")
    print("="*80)
    print(f"{'Strategy':<25} | {'Chunks':<6} | {'Avg Words':<10} | {'Hit Rate@3':<10} | {'MRR@3':<8}")
    print("-"*80)
    for r in results:
        print(
            f"{r['strategy']:<25} | "
            f"{r['chunk_count']:<6} | "
            f"{r['avg_token_count']:<10.1f} | "
            f"{r['hit_rate_at_3']:<10.2%} | "
            f"{r['mrr_at_3']:<8.2%}"
        )
    print("="*80)
    print("\nNote: 'Avg Words' is a word-count proxy estimate for token length.")


if __name__ == "__main__":
    main()
