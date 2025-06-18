import hashlib
import logging
from collections import deque
from pathlib import Path

from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .db import CacheDatabase, CacheFile, get_cache_db, get_vector_store

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".tsv",
    ".xml",
    ".py",
}

IGNORE_DIRECTORIES = {
    ".git",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    ".env",
    ".idea",
    ".vscode",
}


def hash_string(s: str) -> str:
    """Calculate a SHA-256 hash of a string."""
    hasher = hashlib.sha256()
    hasher.update(s.encode("utf-8"))
    return hasher.hexdigest()


def file_content_hash(filepath: Path) -> str:
    """Calculate a hash of the file content."""
    with open(filepath) as f:
        return hash_string(f.read())


def _get_ingested_files(db: CacheDatabase) -> list[CacheFile]:
    """Retrieve a list of ingested files."""
    # This function should interact with the database to retrieve ingested files.
    # For now, it returns an empty list as a placeholder.
    try:
        return db.get_files()
    except Exception as e:
        logger.error(f"Error retrieving ingested files: {e}")
        raise


def _ingest_file(filepath: Path, vector_store: VectorStore) -> list[str]:
    with open(filepath) as fp:
        document = fp.read()
    if not document.strip():
        logger.warning(f"File {filepath} is empty, skipping ingestion.")
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(document)
    chunk_ids = [hash_string(chunk) for chunk in chunks]
    logger.info(f"File {filepath} has been split into {len(chunks)} chunks.")

    vector_store.delete(chunk_ids)
    logger.info(f"Deleted existing chunks for file {filepath} from the vector store.")

    output_ids = vector_store.add_texts(
        texts=chunks,
        ids=chunk_ids,
        metadatas=[{"source": str(filepath)}] * len(chunks),
    )
    logger.info(f"File {filepath} has been ingested into the vector store.")
    logger.debug(f"Output IDs: {output_ids}")
    return output_ids


def ingest_files(files: list[str | Path]):
    logger.info(f"Ingesting files: {files}")

    # Extract metadata from the files.
    files_to_parse = deque([Path(file) for file in files])
    files_and_hashes = []
    while files_to_parse:
        filepath = files_to_parse.popleft()
        if filepath.is_dir() and filepath.name not in IGNORE_DIRECTORIES:
            logger.info(
                f"Found directory: {filepath}, adding its contents to the queue"
            )
            files_to_parse.extend(filepath.iterdir())
        elif filepath.is_file() and filepath.suffix in SUPPORTED_EXTENSIONS:
            file_hash = file_content_hash(filepath)
            files_and_hashes.append((filepath, file_hash))
            logger.info(f"Parsed file: {filepath}")
        else:
            logger.info(f"Ignoring file: {filepath}")

    cache_db = get_cache_db()
    ingested_files = _get_ingested_files(cache_db)
    ingested_files_dict = {file.path: file for file in ingested_files}
    vector_store = get_vector_store()
    for path, content_hash in files_and_hashes:
        # If path is in ingested filed and content hash is not the same,
        # delete the file from the vector store and re-ingest it.
        if (
            path in ingested_files_dict
            and ingested_files_dict[path].content_hash != content_hash
        ):
            logger.info(
                f"File {path} has been updated, re-ingesting it into the vector store."
            )
            chunk_ids = _ingest_file(path, vector_store)
            cache_db.insert_or_update_file(
                path=str(path), content_hash=content_hash, chunk_ids=chunk_ids
            )
        elif path not in ingested_files_dict:
            logger.info(f"File {path} is new, ingesting it into the vector store.")
            chunk_ids = _ingest_file(path, vector_store)
            cache_db.insert_or_update_file(
                path=str(path), content_hash=content_hash, chunk_ids=chunk_ids
            )
        else:
            logger.info(f"File {path} is already ingested and up-to-date, skipping.")

    cache_db.close()
