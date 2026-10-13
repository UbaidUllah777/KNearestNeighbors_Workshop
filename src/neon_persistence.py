
# -*- coding: utf-8 -*-
"""
Persist and verify KNN Heart Disease CSV snapshots
in a remote Neon PostgreSQL database.
"""

from pathlib import Path

import pandas as pd
from sqlalchemy import text


class HeartDiseaseNeonPersistence:
    """Upload and verify Heart Disease datasets in Neon."""

    TABLE_FILES = {
        "knn_heart_raw": "heart_disease_raw.csv",
        "knn_heart_prepared": "heart_disease_prepared.csv",
        "knn_heart_train_cleaned": "heart_train_cleaned.csv",
        "knn_heart_test_cleaned": "heart_test_cleaned.csv",
    }

    EXPECTED_SHAPES = {
        "knn_heart_raw": (303, 14),
        "knn_heart_prepared": (303, 15),
        "knn_heart_train_cleaned": (242, 30),
        "knn_heart_test_cleaned": (61, 30),
    }

    def __init__(self, neon_store, data_dir="data"):
        self.engine = neon_store.engine
        self.data_dir = Path(data_dir)

    def load_local_datasets(self):
        """Load and validate the four local CSV snapshots."""

        datasets = {}

        for table_name, filename in self.TABLE_FILES.items():
            path = self.data_dir / filename

            if not path.is_file():
                raise FileNotFoundError(
                    f"Required CSV not found: {path}"
                )

            df = pd.read_csv(path)

            if df.shape != self.EXPECTED_SHAPES[table_name]:
                raise ValueError(
                    f"Unexpected shape for {filename}: {df.shape}"
                )

            if df.columns.duplicated().any():
                raise ValueError(
                    f"Duplicate columns detected: {filename}"
                )

            datasets[table_name] = df

        # Verify consistency across the prepared and split data
        prepared = datasets["knn_heart_prepared"]
        train = datasets["knn_heart_train_cleaned"]
        test = datasets["knn_heart_test_cleaned"]
        raw = datasets["knn_heart_raw"]

        prepared_ids = set(prepared["record_id"])
        train_ids = set(train["record_id"])
        test_ids = set(test["record_id"])

        if (
            prepared["record_id"].duplicated().any()
            or train["record_id"].duplicated().any()
            or test["record_id"].duplicated().any()
        ):
            raise ValueError("Duplicate record IDs detected.")

        if train_ids & test_ids:
            raise ValueError(
                "Training and testing datasets overlap."
            )

        if train_ids | test_ids != prepared_ids:
            raise ValueError(
                "Train/test record IDs do not match prepared data."
            )

        if not (
            prepared.sort_values("record_id")["heart_disease"]
            .reset_index(drop=True)
            .equals((raw["num"] > 0).astype(int).reset_index(drop=True))
        ):
            raise ValueError(
                "Prepared targets do not match raw targets."
            )

        print("Local CSV validation: PASSED")
        print("Train/test separation: PASSED")

        return datasets

    def upload_and_verify(self):
        """
        Replace only the four dedicated KNN snapshot tables.

        Verify row counts, columns, and values using the
        same database transaction.
        """

        datasets = self.load_local_datasets()
        verified_counts = {}

        with self.engine.begin() as connection:

            for table_name, df in datasets.items():

                # Stable database-only row identifier
                upload_df = df.copy()
                upload_df.insert(
                    0,
                    "snapshot_row",
                    range(len(upload_df))
                )

                # Replace this workshop's snapshot table
                upload_df.to_sql(
                    name=table_name,
                    con=connection,
                    if_exists="replace",
                    index=False,
                    method="multi",
                    chunksize=100,
                )

                # Retrieve rows in their original CSV order
                remote_df = pd.read_sql_query(
                    text(
                        f'SELECT * FROM "{table_name}" '
                        f'ORDER BY "snapshot_row"'
                    ),
                    connection,
                )

                # Verify source and remote data match
                pd.testing.assert_frame_equal(
                    upload_df.reset_index(drop=True),
                    remote_df.reset_index(drop=True),
                    check_dtype=False,
                    check_exact=False,
                    rtol=1e-9,
                    atol=1e-9,
                )

                verified_counts[table_name] = len(remote_df)

                print(
                    f"{table_name}: "
                    f"{len(remote_df)} rows — VERIFIED"
                )

        print("\n=== NEON PERSISTENCE SUMMARY ===")

        for table_name, count in verified_counts.items():
            print(f"{table_name}: {count} rows")

        print(
            "\nTotal stored rows:",
            sum(verified_counts.values())
        )

        print("Database verification: PASSED")

        return verified_counts
