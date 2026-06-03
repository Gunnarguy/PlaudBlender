"""
Migration: Rename pinecone_id to vector_id in segments table.

This migration renames the legacy 'pinecone_id' column to 'vector_id'
to reflect the transition from Pinecone to Qdrant as the primary vector store.

Usage:
    python scripts/migrate_rename_vector_id.py

The migration is idempotent - it will skip if already migrated.
"""

import os
import sys
import sqlite3
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_database_path() -> Path:
    """Get the path to the SQLite database."""
    from src.config import get_settings

    settings = get_settings()
    db_url = settings.database_url

    # Extract path from sqlite:/// URL
    if db_url.startswith("sqlite:///"):
        return Path(db_url[10:])
    else:
        # Default location
        return PROJECT_ROOT / "data" / "brain.db"


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns


def migrate_rename_vector_id():
    """Rename pinecone_id to vector_id in segments table."""
    db_path = get_database_path()

    if not db_path.exists():
        print(f"Database not found at {db_path}")
        print("No migration needed - database will be created with new schema.")
        return

    print(f"Migrating database: {db_path}")

    conn = sqlite3.connect(db_path)

    try:
        # Check if segments table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='segments'"
        )
        if not cursor.fetchone():
            print("Segments table not found - no migration needed.")
            return

        # Check current state
        has_pinecone_id = column_exists(conn, "segments", "pinecone_id")
        has_vector_id = column_exists(conn, "segments", "vector_id")

        if has_vector_id and not has_pinecone_id:
            print("✅ Already migrated - vector_id column exists.")
            return

        if not has_pinecone_id and not has_vector_id:
            print("Neither column exists - adding vector_id column.")
            conn.execute("ALTER TABLE segments ADD COLUMN vector_id TEXT")
            conn.commit()
            print("✅ Added vector_id column.")
            return

        if has_pinecone_id:
            print("Found pinecone_id column - renaming to vector_id...")

            # SQLite doesn't support RENAME COLUMN in older versions,
            # so we need to recreate the table

            # Get existing column info
            cursor = conn.execute("PRAGMA table_info(segments)")
            columns_info = cursor.fetchall()

            # Build new column definitions, replacing pinecone_id with vector_id
            old_cols = []
            new_cols = []
            col_defs = []

            for col in columns_info:
                col_name = col[1]
                col_type = col[2]
                not_null = "NOT NULL" if col[3] else ""
                default = f"DEFAULT {col[4]}" if col[4] is not None else ""
                pk = "PRIMARY KEY" if col[5] else ""

                if col_name == "pinecone_id":
                    # Rename this column
                    old_cols.append("pinecone_id")
                    new_cols.append("vector_id")
                    col_defs.append(
                        f"vector_id {col_type} {not_null} {default} {pk}".strip()
                    )
                else:
                    old_cols.append(col_name)
                    new_cols.append(col_name)
                    col_defs.append(
                        f"{col_name} {col_type} {not_null} {default} {pk}".strip()
                    )

            # Create new table
            create_sql = f"""
                CREATE TABLE segments_new (
                    {', '.join(col_defs)},
                    FOREIGN KEY (recording_id) REFERENCES recordings(id)
                )
            """
            conn.execute(create_sql)

            # Copy data
            copy_sql = f"""
                INSERT INTO segments_new ({', '.join(new_cols)})
                SELECT {', '.join(old_cols)} FROM segments
            """
            conn.execute(copy_sql)

            # Drop old table and rename new one
            conn.execute("DROP TABLE segments")
            conn.execute("ALTER TABLE segments_new RENAME TO segments")

            conn.commit()
            print("✅ Successfully renamed pinecone_id to vector_id!")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Migration: pinecone_id -> vector_id")
    print("=" * 60)
    migrate_rename_vector_id()
    print("=" * 60)
