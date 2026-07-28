"""API Router for triggering document ingestion."""

import shutil
from pathlib import Path
import structlog
from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile, status

from ragchat.ingestion.graph import ingestion_graph

logger = structlog.get_logger(__name__)
router = APIRouter()

# Local upload storage directory
UPLOAD_DIR = Path(__file__).parent.parent.parent.parent / "data" / "uploads"


async def run_ingestion(file_path: Path, chunking_strategy: str):
    """Target function for FastAPI background execution of the Ingestion LangGraph."""
    logger.info("background_ingestion_started", path=str(file_path), strategy=chunking_strategy)
    try:
        # Invoke the compiled LangGraph state machine
        await ingestion_graph.ainvoke({
            "pdf_path": str(file_path),
            "chunking_strategy": chunking_strategy,
        })
        logger.info("background_ingestion_success", path=str(file_path))
    except Exception as exc:
        logger.error("background_ingestion_failed", path=str(file_path), error=str(exc))


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chunking_strategy: str = Form("semantic"),
) -> dict:
    """Upload a PDF document and trigger ingestion asynchronously in the background."""
    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are supported."}

    # Ensure output upload directory exists
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / file.filename

    # Save uploaded file chunk by chunk to prevent loading large files in memory
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info("file_saved_for_ingestion", path=str(file_path))

    # Queue background task
    background_tasks.add_task(run_ingestion, file_path, chunking_strategy)

    return {
        "message": "File upload successful. Ingestion queued in background.",
        "filename": file.filename,
        "chunking_strategy": chunking_strategy,
    }
