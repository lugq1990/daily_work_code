"""Main comparation logic happen here."""
import enum
import numpy as np
import copy
from collections import defaultdict
from numpy.lib.arraysetops import isin
import pandas as pd
from packaging import version
from pandas.core.algorithms import diff


class CompareDataframes:
    @staticmethod
    def _compare_dataframes(df1, df2, cols=None):
        is_same = True

        def _check_version(at_least_version="1.1.0"):
            pd_version = pd.__version__
            if version.parse(pd_version) < version.parse(at_least_version):
                raise RuntimeError(
                    "When try to use pandas `compare` function, should be at least version: {}".format(
                        at_least_version
                    )
                )

        _check_version()

        if cols is not None:
            # in case just need to compare some of columns.
            df1 = df1[cols]
            df2 = df2[cols]

        compare_res = df1.compare(df2)

        # get diff columns name
        diff_cols = list(set([x[0] for x in compare_res.columns]))

        # get diff index
        diff_index = list(compare_res.index)

        # get diff value with each original data rows! Convert it into a JSON should be better
        diff_list_1 = df1.iloc[diff_index, :].to_json(orient="records")
        diff_list_2 = df2.iloc[diff_index, :].to_json(orient="records")

        if len(compare_res) != 0:
            is_same = False

        return (is_same, diff_list_1, diff_list_2, diff_cols, diff_index)

    def compare(self, df_json, df_bq, cols=None):
        """Compare two DFs is same or not?

        Add a logic to what are different records to be returned.

        We need to handle NAN and duplicate records!
        """
        if cols is None:
            cols = df_bq.columns

        # This should base on each row into a tuple to compare.
        is_same = True

        # first let's try to make it into a array, add with column convertion logic in case there is a failure to get data based on columns
        df_json.columns = df_bq.columns
        val1 = df_json[cols].values
        val2 = df_bq[cols].values

        # todo: How to process for shape not equal problem?
        if val1.shape != val2.shape:
            is_same = False

        # just add them with each res dictionary.
        def _get_res_dict(val):
            res = defaultdict(list)

            for i in range(len(val)):
                row = ",".join([str(m) for m in tuple(val[i])])
                if not res.get(row):
                    res[row] = [i]
                else:
                    res[row].append(i)
            return res

        # get each dict
        res1 = _get_res_dict(val1)
        res2 = _get_res_dict(val2)

        # Get common keys
        common_keys = set(res1.keys()) & set(res2.keys())

        # get diff keys
        diff_list_json = list(set(res1.keys()) - common_keys)
        diff_list_bq = list(set(res2.keys()) - common_keys)

        # Get diff index for JSON
        diff_index_json = []
        if diff_list_json:
            is_same = False
            for k in diff_list_json:
                diff_index_json.extend(res1.get(k))
            diff_index_json = sorted(diff_index_json)
            print(
                "Different index: {} for JSON DF".format(
                    "\t".join([str(t) for t in diff_index_json])
                )
            )

        # to get json diff index for BQ
        diff_index_bq = []
        if diff_list_bq:
            is_same = False
            for k in diff_list_bq:
                diff_index_bq.extend(res2.get(k))
            diff_index_bq = sorted(diff_index_bq)
            print(
                "Different index: {} for JSON DF".format(
                    "\t".join([str(t) for t in diff_index_bq])
                )
            )

        # output data just to get diff index
        diff_json = []
        diff_bq = []

        if diff_index_bq:
            diff_bq = df_bq.loc[diff_index_bq, :].to_json(orient="records")

        if diff_index_json:
            diff_json = df_json.loc[diff_index_json, :].to_json(orient="records")

        return is_same, diff_json, diff_bq, diff_index_json, diff_index_bq


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "col1": ["a", "a", "b", "b", "a"],
            "col2": [1.0, 2.0, 3.0, np.nan, 5.0],
            "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        columns=["col1", "col2", "col3"],
    )

    df2 = df.copy()
    df2.loc[0, "col1"] = "c"
    df2.loc[2, "col3"] = 4.0

    com_obj = CompareDataframes()
    print(com_obj.compare(df, df2))

    print("*" * 20)
