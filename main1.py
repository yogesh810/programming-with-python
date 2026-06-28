"""
Main module for the Ideal Function Matching Program.

This program loads training datasets and ideal function datasets,
selects the four best-fit ideal functions using the least-squares criterion,
maps test data points to the chosen ideal functions based on a sqrt(2) deviation
threshold, stores all data in a SQLite database via SQLAlchemy, and visualizes
the results using Bokeh.

Author: Student
Date: 2026
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, Column, Float, Integer, String, MetaData
from sqlalchemy.orm import declarative_base, sessionmaker
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import gridplot, column
from bokeh.models import ColumnDataSource, Legend, Title
from bokeh.palettes import Category10, Spectral6


# Custom Exceptions


class DataLoadError(Exception):
    """
    Custom exception raised when data files cannot be loaded.

    This exception is raised when CSV files are missing, corrupted,
    or contain unexpected data formats.
    """

    def __init__(self, filepath, message="Failed to load data file"):
        """
        Initialize DataLoadError.

        Args:
            filepath (str): Path to the file that failed to load.
            message (str): Description of the error.
        """
        self.filepath = filepath
        self.message = f"{message}: {filepath}"
        super().__init__(self.message)


class MappingError(Exception):
    """
    Custom exception raised when test data mapping encounters issues.

    This exception is raised when the mapping of test data points
    to ideal functions fails due to invalid data or computation errors.
    """

    def __init__(self, message="Error during test data mapping"):
        """
        Initialize MappingError.

        Args:
            message (str): Description of the mapping error.
        """
        self.message = message
        super().__init__(self.message)


# SQLAlchemy ORM Base and Table Models


Base = declarative_base()


class TrainingData(Base):
    """
    SQLAlchemy ORM model for the training data table.

    Stores the combined training data from all four training datasets
    with columns for x and four y-values (y1, y2, y3, y4).

    Attributes:
        id (int): Primary key auto-increment.
        x (float): The x-coordinate value.
        y1 (float): Y-value from training dataset 1.
        y2 (float): Y-value from training dataset 2.
        y3 (float): Y-value from training dataset 3.
        y4 (float): Y-value from training dataset 4.
    """

    __tablename__ = "training_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    x = Column(Float, nullable=False)
    y1 = Column("Y1 (training func)", Float, nullable=False)
    y2 = Column("Y2 (training func)", Float, nullable=False)
    y3 = Column("Y3 (training func)", Float, nullable=False)
    y4 = Column("Y4 (training func)", Float, nullable=False)


class IdealFunction(Base):
    """
    SQLAlchemy ORM model for the ideal functions table.

    Stores all 50 ideal functions loaded from the ideal.csv file.

    Attributes:
        id (int): Primary key auto-increment.
        x (float): The x-coordinate value.
        y1 through y50 (float): Y-values for each of the 50 ideal functions.
    """

    __tablename__ = "ideal_functions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    x = Column(Float, nullable=False)

    # Dynamically define columns y1 to y50
    for i in range(1, 51):
        vars()[f"y{i}"] = Column(f"Y{i} (ideal func)", Float, nullable=False)


class MappedResultRecord(Base):
    """
    SQLAlchemy ORM model for the test data mapping results table.

    Stores the mapping of test data points to their matched ideal functions,
    including the deviation value and the number of the ideal function.

    Note: Named 'MappedResultRecord' (not 'TestResult') to avoid a naming
    conflict with pytest, which attempts to collect any class whose name
    starts with 'Test' as a test suite.

    Attributes:
        id (int): Primary key auto-increment.
        x (float): The x-coordinate of the test point.
        y (float): The y-coordinate of the test point.
        delta_y (float): The deviation between the test and ideal function value.
        ideal_func_no (int): The number of the matched ideal function (or None).
    """

    __tablename__ = "test_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    x = Column("X (test func)", Float, nullable=False)
    y = Column("Y (test func)", Float, nullable=False)
    delta_y = Column("Delta Y (test func)", Float, nullable=True)
    ideal_func_no = Column("No. of ideal func", Integer, nullable=True)


# Data Loader Classes (with Inheritance)


class DataLoader:
    """
    Base class for loading CSV data files using Pandas.

    Provides common functionality for reading CSV files and basic
    data validation. Subclasses specialize for different data types.

    Attributes:
        filepath (str): Path to the CSV file.
        data (pd.DataFrame): Loaded data stored as a Pandas DataFrame.
    """

    def __init__(self, filepath):
        """
        Initialize DataLoader with a file path.

        Args:
            filepath (str): Path to the CSV file to load.
        """
        self.filepath = filepath
        self.data = None

    def load(self):
        """
        Load data from the CSV file.

        Returns:
            pd.DataFrame: The loaded data.

        Raises:
            DataLoadError: If the file cannot be found or loaded.
        """
        try:
            if not os.path.exists(self.filepath):
                raise DataLoadError(self.filepath, "File not found")
            self.data = pd.read_csv(self.filepath)
            self._validate()
            return self.data
        except DataLoadError:
            raise
        except Exception as e:
            raise DataLoadError(self.filepath, str(e))

    def _validate(self):
        """
        Validate the loaded data.

        Base implementation checks that data is not empty.
        Subclasses can override for specific validation.

        Raises:
            DataLoadError: If validation fails.
        """
        if self.data is None or self.data.empty:
            raise DataLoadError(self.filepath, "Loaded data is empty")


class TrainingDataLoader(DataLoader):
    """
    Specialized data loader for training datasets.

    Inherits from DataLoader and adds validation specific to
    training data files (expecting columns 'x' and 'y').
    """

    def _validate(self):
        """
        Validate training data has expected columns 'x' and 'y'.

        Raises:
            DataLoadError: If required columns are missing.
        """
        super()._validate()
        required_cols = {"x", "y"}
        if not required_cols.issubset(set(self.data.columns)):
            raise DataLoadError(
                self.filepath,
                f"Missing required columns. Expected {required_cols}, "
                f"found {set(self.data.columns)}",
            )


class IdealFunctionLoader(DataLoader):
    """
    Specialized data loader for the ideal functions dataset.

    Inherits from DataLoader and validates that the CSV contains
    an 'x' column plus 50 ideal function columns (y1 through y50).
    """

    def _validate(self):
        """
        Validate ideal functions data has 'x' and 50 y-columns.

        Raises:
            DataLoadError: If the expected columns are not present.
        """
        super()._validate()
        if "x" not in self.data.columns:
            raise DataLoadError(self.filepath, "Missing 'x' column")
        if len(self.data.columns) < 51:
            raise DataLoadError(
                self.filepath,
                f"Expected at least 51 columns (x + 50 y-values), "
                f"found {len(self.data.columns)}",
            )


class TestDataLoader(DataLoader):
    """
    Specialized data loader for the test dataset.

    Inherits from DataLoader and validates that the test CSV
    contains the required 'x' and 'y' columns.
    """

    def _validate(self):
        """
        Validate test data has expected columns 'x' and 'y'.

        Raises:
            DataLoadError: If required columns are missing.
        """
        super()._validate()
        required_cols = {"x", "y"}
        if not required_cols.issubset(set(self.data.columns)):
            raise DataLoadError(
                self.filepath,
                f"Missing required columns. Expected {required_cols}, "
                f"found {set(self.data.columns)}",
            )


# Ideal Function Matcher


class IdealFunctionMatcher:
    """
    Selects the best-fit ideal functions and maps test data.

    Uses the least-squares criterion to choose the 4 ideal functions
    that best match the 4 training datasets. Then maps test data points
    to the chosen ideal functions if the deviation does not exceed
    the maximum training deviation multiplied by sqrt(2).

    Attributes:
        training_data (pd.DataFrame): Combined training data (x, y1, y2, y3, y4).
        ideal_data (pd.DataFrame): Ideal functions data (x, y1...y50).
        chosen_functions (dict): Maps training column to chosen ideal column index.
        max_deviations (dict): Max deviation for each chosen ideal function.
        test_results (list): List of test mapping result dictionaries.
    """

    def __init__(self, training_data, ideal_data):
        """
        Initialize IdealFunctionMatcher.

        Args:
            training_data (pd.DataFrame): Combined training data.
            ideal_data (pd.DataFrame): Ideal functions data.
        """
        self.training_data = training_data
        self.ideal_data = ideal_data
        self.chosen_functions = {}
        self.max_deviations = {}
        self.test_results = []

    def select_ideal_functions(self):
        """
        Select the 4 best-fit ideal functions using least squares.

        For each training dataset (y1 to y4), calculates the sum of
        squared deviations against all 50 ideal functions and selects
        the one with the minimum sum.

        Returns:
            dict: Mapping of training column name to chosen ideal function
                  column name and index.

        Raises:
            MappingError: If function selection fails.
        """
        try:
            ideal_y_cols = [col for col in self.ideal_data.columns if col != "x"]
            training_y_cols = [
                col for col in self.training_data.columns if col != "x"
            ]

            for train_col in training_y_cols:
                min_sse = float("inf")
                best_ideal_col = None

                for ideal_col in ideal_y_cols:
                    # Calculate sum of squared errors (SSE)
                    sse = np.sum(
                        (self.training_data[train_col].values
                         - self.ideal_data[ideal_col].values) ** 2
                    )

                    if sse < min_sse:
                        min_sse = sse
                        best_ideal_col = ideal_col

                # Store the chosen ideal function
                self.chosen_functions[train_col] = best_ideal_col

                # Calculate the maximum deviation for this pairing
                deviations = np.abs(
                    self.training_data[train_col].values
                    - self.ideal_data[best_ideal_col].values
                )
                self.max_deviations[best_ideal_col] = np.max(deviations)

                print(
                    f"  Training '{train_col}' -> Ideal '{best_ideal_col}' "
                    f"(SSE: {min_sse:.4f}, Max Dev: {self.max_deviations[best_ideal_col]:.4f})"
                )

            return self.chosen_functions

        except Exception as e:
            raise MappingError(f"Failed to select ideal functions: {str(e)}")

    def map_test_data(self, test_data):
        """
        Map test data points to the chosen ideal functions.

        For each test point (x, y), finds the closest x-value in the
        ideal dataset and checks if the deviation between the test y
        and the ideal y does not exceed max_deviation * sqrt(2).

        Args:
            test_data (pd.DataFrame): Test data with 'x' and 'y' columns.

        Returns:
            list: List of dictionaries with mapping results.

        Raises:
            MappingError: If mapping encounters errors.
        """
        try:
            self.test_results = []

            for _, row in test_data.iterrows():
                test_x = row["x"]
                test_y = row["y"]

                best_match = None
                best_delta = float("inf")

                # Find the closest x-value index in ideal data
                idx = (self.ideal_data["x"] - test_x).abs().idxmin()

                for train_col, ideal_col in self.chosen_functions.items():
                    ideal_y = self.ideal_data.loc[idx, ideal_col]
                    delta = abs(test_y - ideal_y)
                    threshold = self.max_deviations[ideal_col] * math.sqrt(2)

                    if delta <= threshold and delta < best_delta:
                        best_delta = delta
                        # Extract the ideal function number from column name
                        ideal_num = int(ideal_col.replace("y", ""))
                        best_match = {
                            "x": test_x,
                            "y": test_y,
                            "delta_y": round(best_delta, 4),
                            "ideal_func_no": ideal_num,
                        }

                if best_match:
                    self.test_results.append(best_match)
                else:
                    # Test point did not match any ideal function
                    self.test_results.append(
                        {
                            "x": test_x,
                            "y": test_y,
                            "delta_y": None,
                            "ideal_func_no": None,
                        }
                    )

            mapped_count = sum(
                1 for r in self.test_results if r["ideal_func_no"] is not None
            )
            print(
                f"  Mapped {mapped_count}/{len(self.test_results)} test points "
                f"to ideal functions."
            )

            return self.test_results

        except Exception as e:
            raise MappingError(f"Failed to map test data: {str(e)}")


# Database Manager


class DatabaseManager:
    """
    Manages the SQLite database using SQLAlchemy.

    Handles creation of tables and insertion of training data,
    ideal function data, and test mapping results into the database.

    Attributes:
        db_path (str): Path to the SQLite database file.
        engine: SQLAlchemy engine instance.
        Session: SQLAlchemy sessionmaker class.
    """

    def __init__(self, db_path="functions.db"):
        """
        Initialize DatabaseManager and create the database engine.

        Args:
            db_path (str): Path where the SQLite database will be created.
        """
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self.Session = sessionmaker(bind=self.engine)

    def create_tables(self):
        """
        Create all tables defined by SQLAlchemy ORM models.

        Drops existing tables first to ensure a clean database state.
        """
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        print(f"  Database tables created in '{self.db_path}'.")

    def insert_training_data(self, training_df):
        """
        Insert combined training data into the training_data table.

        Args:
            training_df (pd.DataFrame): DataFrame with columns x, y1, y2, y3, y4.
        """
        session = self.Session()
        try:
            for _, row in training_df.iterrows():
                record = TrainingData(
                    x=row["x"],
                    y1=row["y1"],
                    y2=row["y2"],
                    y3=row["y3"],
                    y4=row["y4"],
                )
                session.add(record)
            session.commit()
            print(f"  Inserted {len(training_df)} training data records.")
        except Exception as e:
            session.rollback()
            raise DataLoadError("training_data", f"Failed to insert: {str(e)}")
        finally:
            session.close()

    def insert_ideal_functions(self, ideal_df):
        """
        Insert ideal functions data into the ideal_functions table.

        Args:
            ideal_df (pd.DataFrame): DataFrame with columns x, y1...y50.
        """
        session = self.Session()
        try:
            for _, row in ideal_df.iterrows():
                record = IdealFunction(x=row["x"])
                for i in range(1, 51):
                    setattr(record, f"y{i}", row[f"y{i}"])
                session.add(record)
            session.commit()
            print(f"  Inserted {len(ideal_df)} ideal function records.")
        except Exception as e:
            session.rollback()
            raise DataLoadError("ideal_functions", f"Failed to insert: {str(e)}")
        finally:
            session.close()

    def insert_test_results(self, results):
        """
        Insert test data mapping results into the test_results table.

        Args:
            results (list): List of dictionaries with keys x, y, delta_y,
                           ideal_func_no.
        """
        session = self.Session()
        try:
            for result in results:
                record = MappedResultRecord(
                    x=result["x"],
                    y=result["y"],
                    delta_y=result["delta_y"],
                    ideal_func_no=result["ideal_func_no"],
                )
                session.add(record)
            session.commit()
            print(f"  Inserted {len(results)} test result records.")
        except Exception as e:
            session.rollback()
            raise DataLoadError("test_results", f"Failed to insert: {str(e)}")
        finally:
            session.close()


# Visualizer


class Visualizer:
    """
    Creates Bokeh visualizations for the analysis results.

    Generates interactive HTML plots showing training data vs chosen
    ideal functions, test data mapping results, and combined overviews.

    Attributes:
        training_data (pd.DataFrame): Combined training data.
        ideal_data (pd.DataFrame): Ideal functions data.
        chosen_functions (dict): Mapping of training to ideal function columns.
        test_results (list): Test data mapping results.
    """

    def __init__(self, training_data, ideal_data, chosen_functions, test_results):
        """
        Initialize Visualizer with analysis data.

        Args:
            training_data (pd.DataFrame): Combined training data.
            ideal_data (pd.DataFrame): Ideal functions data.
            chosen_functions (dict): Mapping of training to ideal columns.
            test_results (list): List of test mapping result dictionaries.
        """
        self.training_data = training_data
        self.ideal_data = ideal_data
        self.chosen_functions = chosen_functions
        self.test_results = test_results

    def plot_training_vs_ideal(self, output_path="training_vs_ideal.html"):
        """
        Plot training data against chosen ideal functions.

        Creates a 2x2 grid of plots, each showing one training dataset
        alongside its matched ideal function for visual comparison.

        Args:
            output_path (str): File path for the output HTML file.
        """
        output_file(output_path, title="Training Data vs Chosen Ideal Functions")

        plots = []
        colors = Category10[4]

        for idx, (train_col, ideal_col) in enumerate(self.chosen_functions.items()):
            p = figure(
                title=f"Training '{train_col}' vs Ideal '{ideal_col}'",
                x_axis_label="X",
                y_axis_label="Y",
                width=600,
                height=400,
                tools="pan,wheel_zoom,box_zoom,reset,save,hover",
            )

            # Plot training data as circles
            p.scatter(
                self.training_data["x"],
                self.training_data[train_col],
                size=3,
                color=colors[idx],
                alpha=0.5,
                legend_label=f"Training {train_col}",
            )

            # Plot ideal function as a line
            p.line(
                self.ideal_data["x"],
                self.ideal_data[ideal_col],
                line_width=2,
                color="black",
                legend_label=f"Ideal {ideal_col}",
            )

            p.legend.location = "top_left"
            p.legend.click_policy = "hide"
            plots.append(p)

        grid = gridplot([[plots[0], plots[1]], [plots[2], plots[3]]])
        save(grid)
        print(f"  Saved training vs ideal plot to '{output_path}'.")

    def plot_test_mapping(self, output_path="test_mapping.html"):
        """
        Plot test data mapping results.

        Displays test points colored by their matched ideal function,
        with unmatched points shown in grey. Also overlays the four
        chosen ideal functions as lines.

        Args:
            output_path (str): File path for the output HTML file.
        """
        output_file(output_path, title="Test Data Mapping Results")

        p = figure(
            title="Test Data Mapped to Ideal Functions",
            x_axis_label="X",
            y_axis_label="Y",
            width=1000,
            height=600,
            tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        )

        # Plot chosen ideal functions as lines
        line_colors = Category10[4]
        ideal_cols = list(self.chosen_functions.values())

        for idx, ideal_col in enumerate(ideal_cols):
            p.line(
                self.ideal_data["x"],
                self.ideal_data[ideal_col],
                line_width=2,
                color=line_colors[idx],
                legend_label=f"Ideal {ideal_col}",
                alpha=0.7,
            )

        # Separate mapped and unmapped test points
        mapped = [r for r in self.test_results if r["ideal_func_no"] is not None]
        unmapped = [r for r in self.test_results if r["ideal_func_no"] is None]

        # Plot mapped points grouped by ideal function
        ideal_num_to_color = {}
        for idx, ideal_col in enumerate(ideal_cols):
            ideal_num = int(ideal_col.replace("y", ""))
            ideal_num_to_color[ideal_num] = line_colors[idx]

        for ideal_num, color in ideal_num_to_color.items():
            points = [r for r in mapped if r["ideal_func_no"] == ideal_num]
            if points:
                p.scatter(
                    [pt["x"] for pt in points],
                    [pt["y"] for pt in points],
                    size=8,
                    color=color,
                    alpha=0.8,
                    legend_label=f"Test -> Ideal y{ideal_num}",
                )

        # Plot unmapped points
        if unmapped:
            p.scatter(
                [pt["x"] for pt in unmapped],
                [pt["y"] for pt in unmapped],
                marker="triangle",
                size=8,
                color="grey",
                alpha=0.6,
                legend_label="Unmapped test points",
            )

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        save(p)
        print(f"  Saved test mapping plot to '{output_path}'.")

    def plot_all_data(self, output_path="all_data_overview.html"):
        """
        Create a combined overview plot of all data.

        Shows all four training datasets, the chosen ideal functions,
        and the test data points in a single comprehensive visualization.

        Args:
            output_path (str): File path for the output HTML file.
        """
        output_file(output_path, title="Complete Data Overview")

        p = figure(
            title="Complete Overview: Training Data, Ideal Functions & Test Data",
            x_axis_label="X",
            y_axis_label="Y",
            width=1200,
            height=700,
            tools="pan,wheel_zoom,box_zoom,reset,save,hover",
        )

        colors = Category10[10]

        # Plot all training datasets
        for idx, train_col in enumerate(self.chosen_functions.keys()):
            p.scatter(
                self.training_data["x"],
                self.training_data[train_col],
                size=2,
                color=colors[idx],
                alpha=0.3,
                legend_label=f"Training {train_col}",
            )

        # Plot chosen ideal functions
        for idx, ideal_col in enumerate(self.chosen_functions.values()):
            p.line(
                self.ideal_data["x"],
                self.ideal_data[ideal_col],
                line_width=2,
                color=colors[idx + 4],
                legend_label=f"Ideal {ideal_col}",
            )

        # Plot test data
        mapped = [r for r in self.test_results if r["ideal_func_no"] is not None]
        unmapped = [r for r in self.test_results if r["ideal_func_no"] is None]

        if mapped:
            p.scatter(
                [pt["x"] for pt in mapped],
                [pt["y"] for pt in mapped],
                marker="diamond",
                size=10,
                color="green",
                alpha=0.8,
                legend_label="Mapped test points",
            )

        if unmapped:
            p.scatter(
                [pt["x"] for pt in unmapped],
                [pt["y"] for pt in unmapped],
                marker="triangle",
                size=10,
                color="red",
                alpha=0.8,
                legend_label="Unmapped test points",
            )

        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        save(p)
        print(f"  Saved all data overview plot to '{output_path}'.")


# Main Execution


def main():
    """
    Main function that orchestrates the entire program workflow.

    Steps:
        1. Load training data from four CSV files
        2. Load ideal functions from CSV
        3. Load test data from CSV
        4. Store training and ideal data in SQLite database
        5. Select 4 best ideal functions using least squares
        6. Map test data to chosen ideal functions
        7. Store test mapping results in SQLite database
        8. Generate Bokeh visualizations
    """
    # Define file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ------------------------------------------------------------------
    # FILE NAMES — place all CSV files inside a folder called "datasets"
    # located in the same directory as this main.py file.
    #
    #   datasets/
    #       train1.csv   <- Training dataset 1  (load here)
    #       train2.csv   <- Training dataset 2  (load here)
    #       train3.csv   <- Training dataset 3  (load here)
    #       train4.csv   <- Training dataset 4  (load here)
    #       ideal.csv    <- 50 ideal functions  (load here)
    #       test.csv     <- Test data           (load here)
    # ------------------------------------------------------------------
    datasets_dir = os.path.join(base_dir, "datasets")

    # FILE: train1.csv, train2.csv, train3.csv, train4.csv
    train_files = [
        os.path.join(datasets_dir, f"train{i}.csv") for i in range(1, 5)
    ]

    # FILE: ideal.csv  (contains all 50 ideal functions)
    ideal_file = os.path.join(datasets_dir, "ideal.csv")

    # FILE: test.csv  (contains the test data points)
    test_file = os.path.join(datasets_dir, "test.csv")

    db_path = os.path.join(base_dir, "functions.db")

    print("=" * 60)
    print("  IDEAL FUNCTION MATCHING PROGRAM")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load Training Data
    # LOADS: datasets/train1.csv
    #        datasets/train2.csv
    #        datasets/train3.csv
    #        datasets/train4.csv
    # ------------------------------------------------------------------
    print("\n[1/7] Loading training data...")
    training_dfs = []
    for i, filepath in enumerate(train_files, start=1):
        loader = TrainingDataLoader(filepath)  # loads train1.csv ... train4.csv
        df = loader.load()
        df = df.rename(columns={"y": f"y{i}"})
        training_dfs.append(df)
        print(f"  Loaded train{i}.csv ({len(df)} rows)")

    # Merge all training dataframes on 'x'
    combined_training = training_dfs[0]
    for df in training_dfs[1:]:
        combined_training = combined_training.merge(df, on="x")

    print(f"  Combined training data: {combined_training.shape}")

    # ------------------------------------------------------------------
    # Step 2: Load Ideal Functions
    # LOADS: datasets/ideal.csv
    # ------------------------------------------------------------------
    print("\n[2/7] Loading ideal functions...")
    ideal_loader = IdealFunctionLoader(ideal_file)  # loads ideal.csv
    ideal_data = ideal_loader.load()
    print(f"  Loaded ideal.csv ({ideal_data.shape[0]} rows, "
          f"{ideal_data.shape[1]} columns)")

    # ------------------------------------------------------------------
    # Step 3: Load Test Data
    # LOADS: datasets/test.csv
    # ------------------------------------------------------------------
    print("\n[3/7] Loading test data...")
    test_loader = TestDataLoader(test_file)  # loads test.csv
    test_data = test_loader.load()
    print(f"  Loaded test.csv ({len(test_data)} rows)")

    # ------------------------------------------------------------------
    # Step 4: Store data in SQLite Database
    # ------------------------------------------------------------------
    print("\n[4/7] Creating SQLite database...")
    db_manager = DatabaseManager(db_path)
    db_manager.create_tables()
    db_manager.insert_training_data(combined_training)
    db_manager.insert_ideal_functions(ideal_data)

    # ------------------------------------------------------------------
    # Step 5: Select 4 best ideal functions
    # ------------------------------------------------------------------
    print("\n[5/7] Selecting best ideal functions (least squares)...")
    matcher = IdealFunctionMatcher(combined_training, ideal_data)
    chosen = matcher.select_ideal_functions()

    print("\n  Chosen ideal functions:")
    for train_col, ideal_col in chosen.items():
        print(f"    {train_col} -> {ideal_col}")

    # ------------------------------------------------------------------
    # Step 6: Map test data to chosen ideal functions
    # ------------------------------------------------------------------
    print("\n[6/7] Mapping test data to ideal functions...")
    test_results = matcher.map_test_data(test_data)

    # Store test results in database
    db_manager.insert_test_results(test_results)

    # ------------------------------------------------------------------
    # Step 7: Generate Visualizations
    # ------------------------------------------------------------------
    print("\n[7/7] Generating Bokeh visualizations...")
    visualizer = Visualizer(
        combined_training, ideal_data, chosen, test_results
    )

    vis_dir = base_dir
    visualizer.plot_training_vs_ideal(
        os.path.join(vis_dir, "training_vs_ideal.html")
    )
    visualizer.plot_test_mapping(
        os.path.join(vis_dir, "test_mapping.html")
    )
    visualizer.plot_all_data(
        os.path.join(vis_dir, "all_data_overview.html")
    )

    print("\n" + "=" * 60)
    print("  PROGRAM COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\n  Database: {db_path}")
    print(f"  Visualizations saved in: {vis_dir}")
    print("  Files generated:")
    print("    - training_vs_ideal.html")
    print("    - test_mapping.html")
    print("    - all_data_overview.html")


if __name__ == "__main__":
    main()
