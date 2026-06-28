"""
Unit tests for the Ideal Function Matching Program.

This module contains comprehensive unit tests for the data loading,
ideal function selection, database management, and mapping functionality.

Author: Student
Date: 2026
"""

import os
import sys
import math
import unittest
import tempfile
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect

# Import classes from main module
from main import (
    DataLoader,
    TrainingDataLoader,
    IdealFunctionLoader,
    TestDataLoader,
    IdealFunctionMatcher,
    DatabaseManager,
    DataLoadError,
    MappingError,
    Base,
    TrainingData,
    IdealFunction,
    TestResult,
)


class TestDataLoadError(unittest.TestCase):
    """Test cases for the DataLoadError custom exception."""

    def test_exception_message(self):
        """Test that DataLoadError carries the correct filepath and message."""
        error = DataLoadError("test.csv", "File not found")
        self.assertIn("test.csv", str(error))
        self.assertEqual(error.filepath, "test.csv")

    def test_exception_default_message(self):
        """Test DataLoadError with default message."""
        error = DataLoadError("data.csv")
        self.assertIn("data.csv", str(error))
        self.assertIn("Failed to load data file", error.message)


class TestMappingErrorException(unittest.TestCase):
    """Test cases for the MappingError custom exception."""

    def test_exception_message(self):
        """Test that MappingError carries the correct message."""
        error = MappingError("Computation failed")
        self.assertEqual(str(error), "Computation failed")

    def test_default_message(self):
        """Test MappingError with default message."""
        error = MappingError()
        self.assertIn("Error during test data mapping", str(error))


class TestDataLoader(unittest.TestCase):
    """Test cases for the DataLoader base class."""

    def setUp(self):
        """Create temporary CSV files for testing."""
        self.temp_dir = tempfile.mkdtemp()

        # Create a valid CSV file
        self.valid_csv = os.path.join(self.temp_dir, "valid.csv")
        pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}).to_csv(
            self.valid_csv, index=False
        )

        # Create an empty CSV file
        self.empty_csv = os.path.join(self.temp_dir, "empty.csv")
        pd.DataFrame().to_csv(self.empty_csv, index=False)

    def test_load_valid_file(self):
        """Test loading a valid CSV file."""
        loader = DataLoader(self.valid_csv)
        data = loader.load()
        self.assertIsNotNone(data)
        self.assertEqual(len(data), 3)

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises DataLoadError."""
        loader = DataLoader("nonexistent.csv")
        with self.assertRaises(DataLoadError):
            loader.load()

    def test_load_empty_file(self):
        """Test loading an empty CSV raises DataLoadError."""
        loader = DataLoader(self.empty_csv)
        with self.assertRaises(DataLoadError):
            loader.load()

    def tearDown(self):
        """Clean up temporary files."""
        for f in [self.valid_csv, self.empty_csv]:
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(self.temp_dir)


class TestTrainingDataLoader(unittest.TestCase):
    """Test cases for the TrainingDataLoader class."""

    def setUp(self):
        """Create temporary CSV files for testing."""
        self.temp_dir = tempfile.mkdtemp()

        # Valid training data
        self.valid_csv = os.path.join(self.temp_dir, "train.csv")
        pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}).to_csv(
            self.valid_csv, index=False
        )

        # Invalid training data (missing y column)
        self.invalid_csv = os.path.join(self.temp_dir, "invalid.csv")
        pd.DataFrame({"x": [1, 2, 3], "z": [4, 5, 6]}).to_csv(
            self.invalid_csv, index=False
        )

    def test_load_valid_training_data(self):
        """Test loading valid training data."""
        loader = TrainingDataLoader(self.valid_csv)
        data = loader.load()
        self.assertIn("x", data.columns)
        self.assertIn("y", data.columns)

    def test_load_invalid_columns(self):
        """Test that missing required columns raises DataLoadError."""
        loader = TrainingDataLoader(self.invalid_csv)
        with self.assertRaises(DataLoadError):
            loader.load()

    def test_inheritance(self):
        """Test that TrainingDataLoader is a subclass of DataLoader."""
        self.assertTrue(issubclass(TrainingDataLoader, DataLoader))

    def tearDown(self):
        """Clean up temporary files."""
        for f in [self.valid_csv, self.invalid_csv]:
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(self.temp_dir)


class TestIdealFunctionLoader(unittest.TestCase):
    """Test cases for the IdealFunctionLoader class."""

    def setUp(self):
        """Create temporary CSV files for testing."""
        self.temp_dir = tempfile.mkdtemp()

        # Valid ideal functions data (x + 50 y columns)
        data = {"x": [1, 2, 3]}
        for i in range(1, 51):
            data[f"y{i}"] = [i * 1.0, i * 2.0, i * 3.0]
        self.valid_csv = os.path.join(self.temp_dir, "ideal.csv")
        pd.DataFrame(data).to_csv(self.valid_csv, index=False)

        # Invalid ideal functions data (too few columns)
        self.invalid_csv = os.path.join(self.temp_dir, "invalid_ideal.csv")
        pd.DataFrame({"x": [1, 2], "y1": [3, 4]}).to_csv(
            self.invalid_csv, index=False
        )

    def test_load_valid_ideal_data(self):
        """Test loading valid ideal functions data."""
        loader = IdealFunctionLoader(self.valid_csv)
        data = loader.load()
        self.assertEqual(len(data.columns), 51)

    def test_load_invalid_ideal_data(self):
        """Test that too few columns raises DataLoadError."""
        loader = IdealFunctionLoader(self.invalid_csv)
        with self.assertRaises(DataLoadError):
            loader.load()

    def test_inheritance(self):
        """Test that IdealFunctionLoader is a subclass of DataLoader."""
        self.assertTrue(issubclass(IdealFunctionLoader, DataLoader))

    def tearDown(self):
        """Clean up temporary files."""
        for f in [self.valid_csv, self.invalid_csv]:
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(self.temp_dir)


class TestIdealFunctionMatcher(unittest.TestCase):
    """Test cases for the IdealFunctionMatcher class."""

    def setUp(self):
        """Create sample training and ideal data for testing."""
        # Simple training data: y1 = 2x, y2 = 3x
        x_vals = np.linspace(-5, 5, 20)
        self.training_data = pd.DataFrame(
            {
                "x": x_vals,
                "y1": 2 * x_vals + np.random.normal(0, 0.1, 20),
                "y2": 3 * x_vals + np.random.normal(0, 0.1, 20),
            }
        )

        # Ideal functions: one close to 2x, one close to 3x, others far off
        self.ideal_data = pd.DataFrame(
            {
                "x": x_vals,
                "y1": 2 * x_vals,  # Should match training y1
                "y2": 3 * x_vals,  # Should match training y2
                "y3": 10 * x_vals,  # Far off
                "y4": -5 * x_vals,  # Far off
            }
        )

    def test_select_ideal_functions(self):
        """Test that the correct ideal functions are selected."""
        matcher = IdealFunctionMatcher(self.training_data, self.ideal_data)
        chosen = matcher.select_ideal_functions()

        # y1 in training should map to y1 in ideal (2x)
        self.assertEqual(chosen["y1"], "y1")
        # y2 in training should map to y2 in ideal (3x)
        self.assertEqual(chosen["y2"], "y2")

    def test_number_of_chosen_functions(self):
        """Test that exactly as many functions are chosen as training columns."""
        matcher = IdealFunctionMatcher(self.training_data, self.ideal_data)
        chosen = matcher.select_ideal_functions()
        self.assertEqual(len(chosen), 2)

    def test_max_deviations_calculated(self):
        """Test that max deviations are calculated for each chosen function."""
        matcher = IdealFunctionMatcher(self.training_data, self.ideal_data)
        matcher.select_ideal_functions()
        self.assertEqual(len(matcher.max_deviations), 2)
        for deviation in matcher.max_deviations.values():
            self.assertGreater(deviation, 0)

    def test_map_test_data(self):
        """Test mapping test data to ideal functions."""
        matcher = IdealFunctionMatcher(self.training_data, self.ideal_data)
        matcher.select_ideal_functions()


        x_vals = np.linspace(-5, 5, 20)
        test_x = x_vals[15]
        test_y = 2 * test_x + 0.01  # Very close to ideal y1 = 2x
        test_data = pd.DataFrame({"x": [test_x], "y": [test_y]})
        results = matcher.map_test_data(test_data)

        self.assertEqual(len(results), 1)
        # Point should map to ideal y1 (2x) since deviation is tiny
        self.assertIsNotNone(results[0]["ideal_func_no"])

    def test_unmapped_test_point(self):
        """Test that a far-off test point is not mapped."""
        matcher = IdealFunctionMatcher(self.training_data, self.ideal_data)
        matcher.select_ideal_functions()

        test_data = pd.DataFrame({"x": [1.0], "y": [100.0]})
        results = matcher.map_test_data(test_data)

        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0]["ideal_func_no"])
        self.assertIsNone(results[0]["delta_y"])


class TestDatabaseManager(unittest.TestCase):
    """Test cases for the DatabaseManager class."""

    def setUp(self):
        """Create a temporary database for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_functions.db")
        self.db_manager = DatabaseManager(self.db_path)

    def test_create_tables(self):
        """Test that all required tables are created."""
        self.db_manager.create_tables()

        inspector = inspect(self.db_manager.engine)
        tables = inspector.get_table_names()

        self.assertIn("training_data", tables)
        self.assertIn("ideal_functions", tables)
        self.assertIn("test_results", tables)

    def test_insert_training_data(self):
        """Test inserting training data into the database."""
        self.db_manager.create_tables()

        data = pd.DataFrame(
            {
                "x": [1.0, 2.0],
                "y1": [3.0, 4.0],
                "y2": [5.0, 6.0],
                "y3": [7.0, 8.0],
                "y4": [9.0, 10.0],
            }
        )
        self.db_manager.insert_training_data(data)

        # Verify data was inserted
        session = self.db_manager.Session()
        count = session.query(TrainingData).count()
        session.close()
        self.assertEqual(count, 2)

    def test_insert_test_results(self):
        """Test inserting test results into the database."""
        self.db_manager.create_tables()

        results = [
            {"x": 1.0, "y": 2.0, "delta_y": 0.5, "ideal_func_no": 3},
            {"x": 2.0, "y": 3.0, "delta_y": None, "ideal_func_no": None},
        ]
        self.db_manager.insert_test_results(results)

        session = self.db_manager.Session()
        count = session.query(TestResult).count()
        session.close()
        self.assertEqual(count, 2)

    def tearDown(self):
        """Clean up temporary database."""
        self.db_manager.engine.dispose()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)


class TestMappingCriterion(unittest.TestCase):
    """Test the sqrt(2) deviation mapping criterion."""

    def test_sqrt2_threshold(self):
        """Test that the sqrt(2) factor is correctly applied."""
        max_dev = 1.0
        threshold = max_dev * math.sqrt(2)
        self.assertAlmostEqual(threshold, 1.4142135623730951)

    def test_point_within_threshold(self):
        """Test that a point within the threshold is accepted."""
        max_dev = 2.0
        threshold = max_dev * math.sqrt(2)
        deviation = 2.5  # Less than 2 * sqrt(2) = 2.828
        self.assertTrue(deviation <= threshold)

    def test_point_outside_threshold(self):
        """Test that a point outside the threshold is rejected."""
        max_dev = 2.0
        threshold = max_dev * math.sqrt(2)
        deviation = 3.0  # Greater than 2 * sqrt(2) = 2.828
        self.assertFalse(deviation <= threshold)


if __name__ == "__main__":
    unittest.main()
