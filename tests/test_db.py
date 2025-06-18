from pathlib import Path

import pytest

from pkb.db import CacheDatabase, init_cache_db


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test_pkb.db"

    connection = init_cache_db(
        db_path=db_path, drop=True
    )  # Initialize the database with drop=True for testing
    return CacheDatabase(connection)


class TestCacheDatabase:
    def test_insert_or_update_file(self, db):
        # Test inserting a new file
        db.insert_or_update_file("test_file.txt", "hash1", ["tag1", "tag2"])
        files = db.get_files()
        assert len(files) == 1
        assert files[0].path == Path("test_file.txt")
        assert files[0].content_hash == "hash1"

        # Test updating the same file with a different hash
        db.insert_or_update_file("test_file.txt", "hash2", ["tag1", "tag3"])
        files = db.get_files()
        assert len(files) == 1
        assert files[0].path == Path("test_file.txt")
        assert files[0].content_hash == "hash2"

    def test_get_files(self, db):
        # Test retrieving files from an empty database
        files = db.get_files()
        assert len(files) == 0

        # Insert a file and test retrieval
        db.insert_or_update_file("test_file.txt", "hash1", ["tag1", "tag2"])
        files = db.get_files()
        assert len(files) == 1
        assert files[0].path == Path("test_file.txt")
        assert files[0].content_hash == "hash1"
