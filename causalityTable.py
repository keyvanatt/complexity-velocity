import polars as pl
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np


class CausalityTable():
    def __init__(self, path):
        self.path = path
        self.files = list(Path(path).rglob("*.avro"))

    def load_data(self, date_parsing = True):
        all_data = []
        for file in tqdm(self.files, desc="Loading AVRO files at " + str(self.path)):
            data = pl.read_avro(file)
            date = str(file.name).split("-")[0]
            if date_parsing:
                data = data.with_columns(pl.lit(date).str.strptime(pl.Date, format="%Y%m%d").alias("date"))
            all_data.append(data)
        self.df = pl.concat(all_data)


    def load_one_mounth(self, year, month,date_parsing = True):
        month_path = Path(self.path) / f"year={year}" / f"month={str(month).zfill(2)}"
        month_files = list(Path(month_path).rglob("*.avro"))
        all_data = []
        for file in tqdm(month_files, desc="Loading AVRO files at " + str(month_path)):
            data = pl.read_avro(file)
            date = str(file.name).split("-")[0]
            if date_parsing:
                data = data.with_columns(pl.lit(date).str.strptime(pl.Date, format="%Y%m%d").alias("date"))
            all_data.append(data)
        self.df = pl.concat(all_data)
    