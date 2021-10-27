"""Extraction data comparation logic into a class."""

import numpy as np
import pandas as pd
import warnings

from utils.get_data import BQDataframe, JSONDataframe
from utils.convert_df_util import ConvertDataFrames
from utils.compare import CompareDataframes
from utils.pubsub_util import PublishMessage

warnings.simplefilter('ignore')


class DataCamparationDV:
    """
    Comparation logic for Dashboard and Data warehouse.

    Process step:
        1. Get comparation data
        2. Convert them into same format based on each data type
        3. Comparation between them
    """
    def compare(self, json_obj,  bq_sql):
        df_bq = BQDataframe().get(bq_sql)
        df_json = JSONDataframe().get(json_obj)

        df_json, df_bq = ConvertDataFrames().convert(df_json, df_bq)

        compare_res = CompareDataframes().compare(df_json, df_bq)
        return compare_res
