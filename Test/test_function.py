# # test untitaire
import pandas as pd
import numpy as np
import sys

from numpy import NaN
from pandas import testing as tm

#append the relative location you want to import from
sys.path.append("../py_lib")

import cleanup

def test_missing_values_table():
    df = pd.DataFrame(np.array([[1, NaN], [2, 10], [3, NaN], [4, 100]]), columns=['a', 'b'])
    mis_val_table_ren_columns = cleanup.missing_values_table(df)

    sample = pd.DataFrame(np.array([[2, 50.0]]), columns=['Missing Values', '% of Total Values'])

    tm.assert_series_equal(mis_val_table_ren_columns.reset_index(drop=True)['Missing Values'], sample['Missing Values'].astype(int))
    tm.assert_series_equal(mis_val_table_ren_columns.reset_index(drop=True)['% of Total Values'], sample['% of Total Values'])

def test_transform_treshold():
    df = pd.DataFrame(np.array([[1, 5], [2, 10], [3, 50], [4, 100]]), columns=['a', 'b'])
    new_df = cleanup.transform_treshold(df, 50)
    sample = pd.DataFrame(np.array([[0, 0], [0, 0], [0, 1], [0, 1]]), columns=['a', 'b'])

    tm.assert_series_equal(new_df['a'], sample['a'])
    tm.assert_series_equal(new_df['b'], sample['b'])
