import logging
import os
from dataclasses import dataclass
from pathlib import Path

import duckdb
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

cache_dir = Path(os.environ["HOME"]) / ".cache/pkb"


def init_cache_db(
    db_path: Path | None = None, drop: bool = False
) -> duckdb.DuckDBPyConnection:
    db_path = db_path or cache_dir / "filecache.db"
    if not db_path.parent.exists():
        logger.info(f"No database found, creating one at: {db_path}")
        db_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        logger.info(f"Using existing database at: {db_path}")

    if (wal_file := db_path.with_suffix(".db.wal")).exists():
        wal_file.unlink()

    uri = f"file://{db_path}"
    conn = duckdb.connect(uri)

    if drop:
        logger.info("Dropping existing database tables.")
        conn.execute("DROP TABLE IF EXISTS chunks;")
        conn.execute("DROP TABLE IF EXISTS files;")
        conn.execute("DROP SEQUENCE IF EXISTS files_id_seq;")

    conn.execute(
        """
        CREATE SEQUENCE IF NOT EXISTS files_id_seq;
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY DEFAULT nextval('files_id_seq'),
            path TEXT NOT NULL UNIQUE,
            content_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT NOT NULL,
            file_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (file_id) REFERENCES files(id),
            PRIMARY KEY (id, file_id)
        );
        """
    )
    logger.info("Database initialized successfully.")
    return conn


@dataclass
class CacheFile:
    id: int
    path: Path
    content_hash: str
    created_at: str


@dataclass
class CacheChunk:
    id: str
    file_id: int
    created_at: str


class CacheDatabase:
    def __init__(self, connection: duckdb.DuckDBPyConnection):
        self.conn = connection

    def insert_or_update_file(self, path: str, content_hash: str, chunk_ids: list[str]):
        """Insert a file into the database or update it if it already exists."""
        # Perform delete or update of the files table.
        self.conn.execute(
            """
            INSERT INTO files (path, content_hash)
            VALUES (?, ?)
            ON CONFLICT(path) DO UPDATE SET content_hash = excluded.content_hash
            """,
            (path, content_hash),
        )

        # Get the file ID for the inserted or updated file.
        file_id = self.conn.execute(
            "SELECT id FROM files WHERE path = ?", (path,)
        ).fetchone()
        assert file_id is not None, "File ID should not be None"
        file_id = file_id[0]

        # Delete existing chunks for this file.
        self.conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))

        # Insert new chunks for the file.
        for chunk_id in chunk_ids:
            self.conn.execute(
                "INSERT INTO chunks (id, file_id) VALUES (?, ?)", (chunk_id, file_id)
            )

    def get_files(self) -> list[CacheFile]:
        return [
            CacheFile(id, Path(path), content_hash, created_at)
            for id, path, content_hash, created_at in self.conn.execute(
                "SELECT * FROM files"
            ).fetchall()
        ]

    def get_chunks_for_file(self, file_id: int) -> list[CacheChunk]:
        """Retrieve all chunks for a given file ID."""
        return [
            CacheChunk(id, file_id, created_at)
            for id, file_id, created_at in self.conn.execute(
                "SELECT * FROM chunks WHERE file_id = ?", (file_id,)
            ).fetchall()
        ]

    def close(self):
        self.conn.close()

    def __del__(self):
        """Ensure the database connection is closed when the object is deleted."""
        self.close()


_cache_db = CacheDatabase(init_cache_db())
_vector_store = Chroma(
    collection_name="pkb",
    embedding_function=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        cache_folder=str(cache_dir / "embeddings"),
    ),
    persist_directory=str(cache_dir / "vector_store"),
)


def get_cache_db() -> CacheDatabase:
    """Get a singleton instance of the database connection."""
    return _cache_db


def get_vector_store() -> VectorStore:
    return _vector_store
