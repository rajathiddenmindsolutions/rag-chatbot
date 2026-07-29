"""Docling PDF parsing wrapper."""

import json
from pathlib import Path
import structlog
from docling_core.types.doc import DoclingDocument

logger = structlog.get_logger(__name__)


def parse_pdf_to_docling(pdf_path: Path, output_dir: Path) -> tuple[DoclingDocument, str, Path]:
    """Parse a PDF file using Docling.

    Saves the structured JSON representation to output_dir and returns the DoclingDocument,
    the markdown representation, and the path to the saved JSON.
    """
    logger.info("starting_pdf_parsing", pdf_path=str(pdf_path))
    
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # Configure memory-efficient pipeline options (disable layout/OCR models)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True

    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }

    converter = DocumentConverter(format_options=format_options)
    result = converter.convert(str(pdf_path))

    doc = result.document
    md_text = doc.export_to_markdown()

    # Save JSON output
    json_filename = pdf_path.stem + ".json"
    json_path = output_dir / json_filename

    doc_dict = doc.export_to_dict()
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc_dict, f, indent=2, ensure_ascii=False)

    logger.info("pdf_parsing_complete", pdf_path=str(pdf_path), json_path=str(json_path))
    return doc, md_text, json_path
