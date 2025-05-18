# Python Polars Reference Cheat Sheet

## Table of Contents
- [Introduction](#introduction)
- [Basic Operations](#basic-operations)
- [Series Operations](#series-operations)
- [DataFrame Operations](#dataframe-operations)
- [Data Types](#data-types)
- [Expressions](#expressions)
- [Aggregation Functions](#aggregation-functions)
- [Grouping & Joins](#grouping--joins)
- [Window Functions](#window-functions)
- [Time Series](#time-series)
- [IO Operations](#io-operations)
- [Performance Tips](#performance-tips)
- [Conversion With Pandas](#conversion-with-pandas)
- [Advanced Techniques](#advanced-techniques)

## Introduction

Polars is a lightning-fast DataFrame library written in Rust with Python bindings, designed for performance and efficiency with large datasets.

```python
import polars as pl
```

**Key Advantages**:
- Lazy evaluation for optimized execution plans
- Streaming functionality for processing data larger than RAM
- Multi-threaded processing by default
- Expression-based API for concise, powerful operations
- Strong type safety

## Basic Operations

### Creating DataFrames

```python
# From dictionary
df = pl.DataFrame({
    "A": [1, 2, 3],
    "B": ["a", "b", "c"],
    "C": [True, False, True]
})

# From list of dictionaries
df = pl.DataFrame([
    {"A": 1, "B": "a", "C": True},
    {"A": 2, "B": "b", "C": False},
    {"A": 3, "B": "c", "C": True}
])

# From NumPy array
import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6]])
df = pl.DataFrame(arr, schema=["A", "B", "C"])

# Empty DataFrame with schema
df = pl.DataFrame(schema={"A": pl.Int64, "B": pl.Utf8, "C": pl.Boolean})

# Range of values
df = pl.DataFrame({
    "values": range(10)
})
```

### Creating Series

```python
# Basic Series
s = pl.Series("column_name", [1, 2, 3, 4])

# With specific data type
s = pl.Series("column_name", [1, 2, 3, 4], dtype=pl.Float64)

# From numpy array
s = pl.Series("column_name", np.array([1, 2, 3, 4]))
```

### Basic Inspection

```python
# General info
df.schema
df.dtypes
df.shape
df.columns
df.width
df.height
df.estimated_size()

# Content inspection
df.head(5)
df.tail(5)
df.sample(5)
df.describe()
df.glimpse()
df.show()  # Pretty print the DataFrame

# For Series
s.name
s.dtype
s.len()
s.to_list()
```

## Series Operations

### Basic Operations

```python
# Accessing elements
s[0]  # First element
s[-1]  # Last element
s.item()  # Get single value (if Series has only one element)

# Slicing
s[1:4]  # Elements 1 through 3
s[:5]   # First 5 elements
s[-5:]  # Last 5 elements

# Basic operations
s1 + s2  # Element-wise addition
s * 2    # Multiply all elements by 2
s**2     # Square each element

# Comparison
s > 5
s.is_in([1, 2, 3])
s.is_null()
s.is_not_null()
s.is_unique()
s.is_duplicated()
```

### Transformation Methods

```python
# Type conversions
s.cast(pl.Float64)
s.to_numpy()
s.to_list()
s.to_pandas()

# Mathematical operations
s.abs()
s.clip(min_val, max_val)
s.sin()
s.cos()
s.log()
s.exp()
s.ceil()
s.floor()
s.round(decimals=2)

# String operations
s.str.to_lowercase()
s.str.to_uppercase()
s.str.strip()
s.str.replace(pattern, replacement)
s.str.contains(pattern)
s.str.extract(pattern, group_index=0)
s.str.split(by=" ")
s.str.slice(start, length)
s.str.length()

# Date/time operations
s.dt.year()
s.dt.month()
s.dt.day()
s.dt.hour()
s.dt.strftime(format="%Y-%m-%d")
```

## DataFrame Operations

### Selection

```python
# Column selection
df.select("column_name")
df.select(["col1", "col2"])
df.select(pl.col("col1"), pl.col("col2") * 2)

# Using $ notation for expressions
df.select(pl.col("$foo"))  # Selects column named "foo"

# Row selection
df.filter(pl.col("A") > 5)
df.filter((pl.col("A") > 5) & (pl.col("B") == "x"))
df.head(5)
df.tail(5)
df.sample(n=10, seed=42)

# Combined selection
df.filter(pl.col("A") > 5).select(["B", "C"])

# Using with_columns to keep all columns and add or modify selected columns
df.with_columns(
    pl.col("A").alias("A_squared"),
    (pl.col("B") * 2).alias("B_doubled")
)

# Using item() to get a single scalar value
df.select(pl.sum("value")).item()
```

### Manipulation

```python
# Adding columns
df.with_columns(
    pl.lit(5).alias("new_column"),
    (pl.col("A") * 2).alias("A_doubled")
)

# Renaming columns
df.rename({"old_name": "new_name"})

# Dropping columns
df.drop(["col1", "col2"])

# Sorting
df.sort("column_name")
df.sort("column_name", descending=True)
df.sort(["col1", "col2"], descending=[True, False])

# Filling null values
df.fill_null(0)
df.fill_null(strategy="mean")

# Unique rows
df.unique()
df.unique(subset=["col1", "col2"])

# Exploding lists into rows
df.explode("list_column")

# Melting (wide to long format)
df.melt(id_vars=["id"], value_vars=["val1", "val2"])

# Pivoting (long to wide format)
df.pivot(index="id", columns="category", values="value")
```

### Lazy Execution

```python
# Creating a lazy DataFrame
lazy_df = df.lazy()
lazy_df = pl.scan_csv("large_file.csv")

# Building a query
query = (
    lazy_df
    .filter(pl.col("value") > 100)
    .group_by("category")
    .agg(pl.sum("amount"))
    .sort("amount", descending=True)
)

# Executing the query
result = query.collect()

# Optimized execution
result = query.sink_optimized().collect()

# Streaming execution (for large datasets)
result = query.collect(streaming=True)

# Examining execution plan
print(query.describe_optimized_plan())
```

## Data Types

```python
# Primitive types
pl.Int8, pl.Int16, pl.Int32, pl.Int64
pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64
pl.Float32, pl.Float64
pl.Boolean
pl.Utf8  # String type
pl.Binary  # Raw binary data

# Date/Time types
pl.Date
pl.Datetime
pl.Time
pl.Duration

# Nested types
pl.List
pl.Struct
pl.Array

# Null type
pl.Null

# Checking and converting types
df.dtypes
df = df.with_columns(pl.col("A").cast(pl.Float64))
```

## Expressions

### Creation Functions

```python
# Basic expressions
pl.col("column_name")  # Reference a column
pl.lit(5)  # Literal value
pl.when(pl.col("x") > 0).then(1).otherwise(0)  # Conditional expression

# String expressions
pl.col("text").str.contains("pattern")
pl.col("text").str.to_lowercase()

# Date expressions
pl.col("date").dt.year()
pl.col("date").dt.month()

# List expressions
pl.col("list").list.len()
pl.col("list").list.sum()

# Struct expressions
pl.col("struct").struct.field("nested_field")
```

### Common Expression Operations

```python
# Arithmetic
pl.col("A") + pl.col("B")
pl.col("A") * 5
pl.col("A").sum()

# Comparisons
pl.col("A") > 5
pl.col("A").is_between(1, 10)
pl.col("A").is_in([1, 2, 3])

# Null handling
pl.col("A").is_null()
pl.col("A").is_not_null()
pl.col("A").fill_null(0)

# String operations
pl.col("text").str.contains("pattern")
pl.col("text").str.split(" ")

# List operations
pl.col("list").list.get(0)  # Get first element
pl.col("list").list.join("-")  # Join elements with separator

# Struct operations
pl.col("struct").struct.field("field_name")
```

### Aggregation Expressions

```python
pl.col("value").sum()
pl.col("value").mean()
pl.col("value").min()
pl.col("value").max()
pl.col("value").std()
pl.col("value").var()
pl.col("value").count()
pl.col("value").n_unique()
pl.col("value").first()
pl.col("value").last()
pl.col("value").median()
pl.col("value").quantile(0.75)

# String aggregations
pl.col("text").str.concat("-")

# List aggregations
pl.col("list").arr.max()
```

## Aggregation Functions

```python
# Basic aggregations
df.select(
    pl.sum("value"),
    pl.mean("value"),
    pl.min("value"),
    pl.max("value"),
    pl.std("value"),
    pl.var("value"),
    pl.count("value"),
    pl.n_unique("value")
)

# Multiple columns
df.select([
    pl.all().sum(),  # Sum all numeric columns
    pl.all().mean()  # Mean of all numeric columns
])

# Custom aggregations
df.select(
    pl.col("value").map(lambda x: x**2).mean().alias("mean_squared")
)

# Group by aggregations
df.group_by("category").agg(
    pl.sum("value").alias("total"),
    pl.mean("value").alias("average"),
    pl.count("value").alias("count")
)
```

## Grouping & Joins

### Group By Operations

```python
# Basic group by
df.group_by("category").agg(
    pl.sum("amount").alias("total_amount"),
    pl.count("id").alias("count")
)

# Multi-column group by
df.group_by(["category", "region"]).agg(
    pl.col("amount").sum(),
    pl.col("amount").mean(),
    pl.col("id").count()
)

# Dynamic group by
df.group_by_dynamic(
    "date_col",
    every="1w",  # Weekly groups
    closed="left",
    label="left"
).agg(
    pl.sum("amount")
)

# Rolling group by
df.sort("date").rolling(
    index_column="date",
    period="5d",  # 5-day window
    closed="right"
).agg(
    pl.mean("value").alias("rolling_mean")
)
```

### Join Operations

```python
# Inner join
df1.join(df2, on="key", how="inner")

# Left join
df1.join(df2, on="key", how="left")

# Right join
df1.join(df2, on="key", how="right")

# Outer join
df1.join(df2, on="key", how="outer")

# Cross join
df1.join(df2, how="cross")

# Join on multiple columns
df1.join(df2, on=["key1", "key2"])

# Join with different column names
df1.join(df2, left_on="df1_key", right_on="df2_key")

# Asof join (for time series data)
df1.join_asof(
    df2,
    left_on="time1",
    right_on="time2",
    strategy="backward",  # or "forward"
    tolerance="5m"  # 5 minutes tolerance
)
```

## Window Functions

```python
# Basic window functions
df.with_columns(
    pl.col("value").sum().over("category").alias("category_total"),
    pl.col("value").mean().over("category").alias("category_mean")
)

# Multiple partitions
df.with_columns(
    pl.col("value").sum().over(["category", "region"]).alias("group_total")
)

# With sorting
df.with_columns(
    pl.col("value").cum_sum().over("category").sort("date").alias("running_total")
)

# Rolling window
df.with_columns(
    pl.col("value")
      .mean()
      .over("category")
      .sort("date")
      .rolling(window_size=3)
      .alias("moving_avg")
)

# Ranked window functions
df.with_columns(
    pl.col("value").rank().over("category").alias("rank"),
    pl.col("value").pct_rank().over("category").alias("percentile_rank"),
    pl.col("value").dense_rank().over("category").alias("dense_rank"),
    pl.col("value").row_number().over("category").sort("value").alias("row_num")
)

# Offsets in windows
df.with_columns(
    pl.col("value").shift(1).over("category").sort("date").alias("previous_value"),
    pl.col("value").shift(-1).over("category").sort("date").alias("next_value")
)
```

## Time Series

### Creating Date/Time Data

```python
# From strings
df = df.with_columns(
    pl.col("date_str").str.to_date("%Y-%m-%d").alias("date"),
    pl.col("datetime_str").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("datetime")
)

# From components
df = df.with_columns(
    pl.date(pl.col("year"), pl.col("month"), pl.col("day")).alias("date")
)

# Date range
dates = pl.date_range(
    low=datetime(2022, 1, 1),
    high=datetime(2022, 12, 31),
    interval="1d"
)
```

### Time Series Operations

```python
# Extracting components
df = df.with_columns(
    pl.col("date").dt.year().alias("year"),
    pl.col("date").dt.month().alias("month"),
    pl.col("date").dt.day().alias("day"),
    pl.col("datetime").dt.hour().alias("hour"),
    pl.col("datetime").dt.minute().alias("minute"),
    pl.col("datetime").dt.second().alias("second"),
    pl.col("date").dt.weekday().alias("weekday"),
    pl.col("date").dt.week().alias("week_num")
)

# Formatting
df = df.with_columns(
    pl.col("date").dt.strftime("%Y-%m-%d").alias("date_str"),
    pl.col("datetime").dt.strftime("%Y-%m-%d %H:%M:%S").alias("datetime_str")
)

# Time arithmetic
df = df.with_columns(
    (pl.col("datetime") + timedelta(days=1)).alias("next_day"),
    (pl.col("datetime") - timedelta(hours=2)).alias("two_hours_ago")
)

# Truncation
df = df.with_columns(
    pl.col("datetime").dt.truncate("1d").alias("day_start"),
    pl.col("datetime").dt.truncate("1h").alias("hour_start"),
    pl.col("datetime").dt.truncate("1mo").alias("month_start")
)

# Time zones
df = df.with_columns(
    pl.col("datetime").dt.replace_time_zone("UTC").alias("utc_time"),
    pl.col("datetime").dt.convert_time_zone("America/New_York").alias("ny_time")
)
```

### Time Windows

```python
# Dynamic grouping for time series
result = df.group_by_dynamic(
    "timestamp",
    every="1h",    # 1-hour windows
    period="3h",   # 3-hour windows (sliding)
    offset="30m",  # Offset by 30 minutes
    closed="left", # Include left boundary, exclude right
    label="left"   # Use left boundary for label
).agg(
    pl.col("value").mean().alias("hourly_avg")
)

# Common interval strings
# "1ns", "1us", "1ms" - Nanoseconds, microseconds, milliseconds
# "1s", "1m", "1h" - Seconds, minutes, hours
# "1d", "1w", "1mo", "1y" - Days, weeks, months, years
```

## IO Operations

### CSV

```python
# Reading
df = pl.read_csv("data.csv")

# With options
df = pl.read_csv(
    "data.csv",
    sep=",",
    has_header=True,
    skip_rows=1,
    columns=["col1", "col2"],
    dtypes={"col1": pl.Int64, "col2": pl.Utf8},
    null_values=["NA", "null"],
    low_memory=True
)

# Writing
df.write_csv("output.csv")
df.write_csv("output.csv", sep="|", include_header=True)

# Lazy reading
lazy_df = pl.scan_csv("large_file.csv")
```

### Parquet

```python
# Reading
df = pl.read_parquet("data.parquet")

# With options
df = pl.read_parquet(
    "data.parquet",
    columns=["col1", "col2"],
    use_pyarrow=True,
    memory_map=True
)

# Writing
df.write_parquet("output.parquet")
df.write_parquet(
    "output.parquet",
    compression="snappy",
    statistics=True
)

# Lazy reading
lazy_df = pl.scan_parquet("large_file.parquet")
```

### JSON

```python
# Reading
df = pl.read_json("data.json")

# Writing
df.write_json("output.json")
df.write_json("output.json", pretty=True)
```

### Excel

```python
# Reading
df = pl.read_excel("data.xlsx")
df = pl.read_excel("data.xlsx", sheet_name="Sheet1")

# Writing
df.write_excel("output.xlsx")
```

### Other Formats

```python
# Arrow
df = pl.from_arrow(arrow_table)
arrow_table = df.to_arrow()

# NumPy
df = pl.from_numpy(numpy_array, schema=["col1", "col2"])
numpy_array = df.to_numpy()

# Dict
df = pl.DataFrame(dict_data)
dict_data = df.to_dict()

# Apache Avro
df = pl.read_avro("data.avro")
df.write_avro("output.avro")

# IPC/Feather
df = pl.read_ipc("data.feather")
df.write_ipc("output.feather")
```

## Performance Tips

### Memory Management

```python
# Check memory usage
df.estimated_size()

# Optimize string memory usage
df = df.with_columns(pl.col("str_col").str.to_categorical())

# Use appropriate data types
df = df.with_columns([
    pl.col("int_col").cast(pl.Int32),    # If values fit in 32 bits
    pl.col("float_col").cast(pl.Float32) # If precision is sufficient
])
```

### Execution Optimization

```python
# Use lazy execution for complex operations
result = (
    df.lazy()
    .filter(pl.col("value") > 0)
    .group_by("category")
    .agg(pl.col("value").sum())
    .collect()
)

# Streaming mode for large datasets
result = (
    pl.scan_csv("huge_file.csv")
    .filter(pl.col("value") > 1000)
    .collect(streaming=True)
)

# Parallel execution settings
import polars as pl
pl.Config.set_global_config(
    thread_count=8,       # Number of threads to use
    streaming_chunk_size=1_000_000  # Rows per chunk in streaming mode
)
```

### Expression Optimization

```python
# Use vectorized operations
# Good:
df = df.with_columns(pl.col("value") * 2)

# Avoid Python UDFs when possible
# Less efficient:
df = df.with_columns(pl.col("value").map(lambda x: x * 2))

# Use predicate pushdown
result = (
    df.lazy()
    .filter(pl.col("category") == "A")  # This filter will be pushed down
    .select(pl.col("value").sum())
    .collect()
)
```

## Conversion With Pandas

```python
# From Pandas to Polars
import pandas as pd
pandas_df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
polars_df = pl.from_pandas(pandas_df)

# From Polars to Pandas
pandas_df = polars_df.to_pandas()

# With Arrow as intermediary (can be more efficient)
arrow_table = pandas_df.to_arrow()
polars_df = pl.from_arrow(arrow_table)
```

## Advanced Techniques

### Custom Functions

```python
# Using Python UDFs (slower but flexible)
df = df.with_columns(
    pl.col("text").map(lambda s: s.upper()).alias("upper_text")
)

# Vectorized functions (faster)
df = df.with_columns(
    (pl.col("A") * pl.col("B") + pl.col("C")).alias("custom_calc")
)

# Register custom function with expressions
@pl.api.register_expr_namespace("custom")
class CustomNamespace:
    def __init__(self, expr):
        self._expr = expr
    
    def double(self) -> pl.Expr:
        return self._expr * 2

# Use custom namespace
df = df.with_columns(
    pl.col("value").custom.double().alias("doubled")
)
```

### Working with Missing Data

```python
# Detecting missing values
df = df.with_columns(
    pl.col("value").is_null().alias("is_missing"),
    pl.col("value").is_not_null().alias("is_present")
)

# Counting missing values
missing_counts = df.null_count()

# Filtering rows with missing values
df_complete = df.filter(pl.all_horizontal(pl.all().is_not_null()))
df_with_nulls = df.filter(pl.any_horizontal(pl.all().is_null()))

# Filling missing values
df = df.fill_null(strategy="forward")  # Forward fill
df = df.fill_null(strategy="backward")  # Backward fill
df = df.fill_null(strategy="min")  # Fill with minimum value
df = df.fill_null(strategy="max")  # Fill with maximum value
df = df.fill_null(strategy="mean")  # Fill with mean value
df = df.fill_null(strategy="zero")  # Fill with zeros
df = df.fill_null(pl.lit(0))  # Fill with specific value

# Fill nulls with different values by column
df = df.with_columns(
    pl.when(pl.col("A").is_null())
      .then(pl.lit(0))
      .otherwise(pl.col("A"))
      .alias("A"),
    pl.when(pl.col("B").is_null())
      .then(pl.lit("unknown"))
      .otherwise(pl.col("B"))
      .alias("B")
)
```

### List and Struct Operations

```python
# Working with list columns
df = df.with_columns(
    pl.col("list_col").list.get(0).alias("first_element"),
    pl.col("list_col").list.len().alias("list_size"),
    pl.col("list_col").list.sum().alias("list_sum"),
    pl.col("list_col").list.sort().alias("sorted_list"),
    pl.col("list_col").list.reverse().alias("reversed_list"),
    pl.col("list_col").list.unique().alias("unique_elements"),
    pl.col("list_col").list.eval(pl.element() * 2).alias("doubled_elements")
)

# Working with struct columns
df = df.with_columns(
    pl.col("struct_col").struct.field("nested_field").alias("extracted_field"),
    pl.struct(
        pl.col("A").alias("field1"),
        pl.col("B").alias("field2")
    ).alias("new_struct")
)

# Unnesting lists and structs
df_exploded = df.explode("list_column")
df_unnested = df.unnest("struct_column")
```

### Working with ROW-wise Operations

```python
# Apply a function to each row
df = df.with_columns(
    pl.struct(["A", "B", "C"])
      .apply(lambda row: row["A"] + row["B"] + row["C"])
      .alias("row_sum")
)

# Horizontal aggregations
df = df.with_columns(
    pl.max_horizontal(pl.col("A"), pl.col("B"), pl.col("C")).alias("max_value"),
    pl.min_horizontal(pl.col("A"), pl.col("B"), pl.col("C")).alias("min_value"),
    pl.sum_horizontal(pl.col("A"), pl.col("B"), pl.col("C")).alias("sum_value")
)

# Map multiple columns together
df = df.with_columns(
    pl.map_horizontal(["A", "B"], lambda a, b: a / b if b != 0 else None).alias("A_div_B")
)
```

### Advanced Filtering

```python
# Complex predicates
df.filter(
    (pl.col("A") > 5) & 
    (pl.col("B").is_in(["x", "y", "z"])) & 
    (~pl.col("C").is_null())
)

# Pattern matching
df.filter(pl.col("text").str.contains("pattern"))

# Range filters
df.filter(pl.col("value").is_between(10, 20))

# Filtering with window functions
df.filter(
    pl.col("value") > pl.col("value").mean().over("category")
)

# Top N per group
df.filter(
    pl.col("rank").over("category").sort("value", descending=True) <= 3
)
```

### Dynamic Column Operations

```python
# Select columns by pattern
df.select(pl.col("^col_[0-9]$"))  # Regex pattern

# Apply function to multiple columns
df.with_columns(
    pl.col(pl.Float64).log().prefix("log_")
)

# Operate on columns matching a predicate
df.with_columns(
    pl.col(pl.selector.dtype(pl.Utf8)).str.to_uppercase().suffix("_upper")
)

# Use selectors
numeric_cols = pl.selector.dtype(pl.NUMERIC_DTYPES)
df.with_columns(
    pl.col(numeric_cols).fill_null(0)
)
```