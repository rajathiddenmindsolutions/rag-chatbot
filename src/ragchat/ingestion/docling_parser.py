"""Docling PDF parsing wrapper."""

import json
from pathlib import Path
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
    pipeline_options.do_table_structure = False
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    # Run conversion
    result = converter.convert(pdf_path)
    doc = result.document
    
    # Export to markdown
    markdown_text = doc.export_to_markdown()
    
    # Save structured JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{pdf_path.stem}_docling.json"
    
    # DoclingDocument supports serialization
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc.model_dump(), f, ensure_ascii=False, indent=2)
        
    logger.info("pdf_parsing_completed", json_path=str(json_path))
    return doc, markdown_text, json_path
