
# -*- coding: utf-8 -*-
"""Neon PostgreSQL persistence component."""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


class NeonStore:
    """Manage a remote PostgreSQL database connection."""

    def __init__(self):
        # Load the project root .env file
        project_root = Path(__file__).resolve().parent.parent
        load_dotenv(project_root / ".env")

        database_url = os.getenv("DATABASE_URL")

        if not database_url:
            raise ValueError(
                "DATABASE_URL is missing from the .env file."
            )

        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 10},
        )

    def check_connection(self):
        """Verify connectivity without modifying database data."""

        with self.engine.connect() as connection:
            result = connection.execute(
                text("SELECT current_database(), 1")
            ).one()

        print("Neon PostgreSQL connection: SUCCESS")
        print("Database:", result[0])
        print("Connection test:", "PASSED" if result[1] == 1 else "FAILED")

        return result[1] == 1

    def close(self):
        """Dispose of database connection resources."""
        self.engine.dispose()
