"""Main compare logic happen here."""
import pandas as pd
import re
import string
from decimal import Decimal, localcontext, ROUND_HALF_UP


class ConvertDataFrames:
    def __init__(self) -> None:
        # placeholder for object to be used.
        self.df_json = None
        self.df_bq = None

    def get_same_diff_type_cols(self):
        # get two dataframe's data types, and get the same data type's columns
        json_dtype = dict(self.df_json.dtypes)
        bq_dtype = dict(self.df_bq.dtypes)

        same_dtype_cols = []
        other_dtype_cols = []
        for k, _ in json_dtype.items():
            if json_dtype[k] == bq_dtype[k]:
                same_dtype_cols.append(k)
            else:
                other_dtype_cols.append(k)

        return same_dtype_cols, other_dtype_cols

    def convert_date_cols(self):
        """To convert datetime columns to be same format

        If both of them are date type, then we could use pandas.to_datetime try to convert them into a normal datetime, 
        and they will be same,otherwise we will get error then should be False returned.
        """        
        # change here, in case that there are other types of datetime.
        # with func: `select_dtype` for robost selection of date
        sati_date_types = ['datetime64', 'timedelta64', 'datetime64[ns]']
        
        tmp_date_df_bq = self.df_bq.select_dtypes(sati_date_types)

        if tmp_date_df_bq.shape[0] == 0:
            # there isn't any datetime columns, just return
            return 
        
        sati_date_cols = list(tmp_date_df_bq.columns)
        print("Get columns: {} with datetime to process.".format('\t'.join(sati_date_cols)))

        self.df_bq[sati_date_cols] = self.df_bq[sati_date_cols].apply(pd.to_datetime)
        self.df_json[sati_date_cols] = self.df_json[sati_date_cols].apply(pd.to_datetime)

    # Here should add a function that convert float, the reason add this function
    # is that for 0.5 with only rounding with return 0, in fact we want 1 to be returned.
    @staticmethod
    def convert_float_round(x, n_keep_digists=4):
        with localcontext() as ctx:
            ctx.rounding = ROUND_HALF_UP
            return float(round(Decimal(x), n_keep_digists))

    def convert_percen_cols(self,
                        float_round_estimation = 4,
                        per_threshould = .9):
        """This is a Pipeline, DF's dtype will be same after we have processed.

        Args:
            df_json ([type]): [description]
            df_bq ([type]): [description]
            float_round_estimation ([type], optional): [description]. Defaults to 4.
            per_threshould ([type], optional): [description]. Defaults to .9.

        Returns:
            [type]: [description]
        """
        same_dtype_cols, _ = self.get_same_diff_type_cols()
        other_not_sati_cols = list(set(list(self.df_bq.columns)) - set(same_dtype_cols))
        
        if len(other_not_sati_cols) == 0:
            print("There isn't not others type of columns in BigQuery DataFrame.")
            return 
        
        
        other_sati_json_df = self.df_json[other_not_sati_cols].astype(str)
        other_sati_bq_df = self.df_bq[other_not_sati_cols]

        # Loop each columns to get percentage columns.
        # todo: for column: 1. remove nan; 2. map(lambda x: '%' in x).reduce(sum)
        percen_cols = []
        for col in other_sati_json_df.columns:
            per_num = other_sati_json_df[col].map(lambda x: True if "%" in x else False).sum()
            null_num = other_sati_json_df[col].isnull().sum()
            if per_num:
                if null_num:
                    if per_num / (null_num + per_num) >= per_threshould:
                        percen_cols.append(col)
                else:
                    if per_num / len(other_sati_json_df) >= per_threshould:
                        percen_cols.append(col)
        
        if len(percen_cols) == 0:
            print("There isn't percentage columns in JSON DataFrame.")
            return 
        
        print("Get columns: {} as Percentage column to process.".format('\t'.join(percen_cols)))
            
        # If we have get percentage columns, then need to convert them into float
        other_sati_json_df[percen_cols] = other_sati_json_df[percen_cols].applymap(lambda x: float(x.replace('%', ''))/100)
        
        # Key notes here: WE SHOULDN'T COMPARE FLOAT, SHOULD CONVERT INTO STRING!
        # convert BQ df either, so could compare easy...Let's just hard-code this for 4-digits to keep
        per_convert_json = other_sati_json_df[percen_cols].applymap(lambda x: "%.4f" %  round(x, float_round_estimation))
        
        # Here I add with rounding logic that will convert float into rounding logic, but only for BQ DF only!
        other_sati_bq_df[percen_cols] = other_sati_bq_df[percen_cols].applymap(lambda x: self.convert_float_round(x))
        
        per_convert_bq = other_sati_bq_df[percen_cols].applymap(lambda x: "%.4f" % round(x, float_round_estimation))

        # write these columns back with these new DFs.
        self.df_json[percen_cols] = per_convert_json
        self.df_bq[percen_cols] = per_convert_bq
        
        return 

    def convert_other_spe_cols(self):
        """As this is a pipeline, after each step process, then we will convert diff columns into same.

        So Here if we need to try with other columns types, then we could just try to set data from original.

        Args:
            df_json ([type]): [description]
            df_bq ([type]): [description]

        Returns:
            [type]: [description]
        """
        same_dtype_cols, _ = self.get_same_diff_type_cols()
        other_str_cols = list(set(list(self.df_bq.columns)) - set(same_dtype_cols))
        special_characters = re.escape(string.punctuation)

        def remove_spe_cha(x):
            return re.sub(r"[" + special_characters + "]", "", str(x))

        if not other_str_cols:
            print("There isn't any other String columns to process.")
            return 
        
        self.df_bq[other_str_cols] = self.df_bq[other_str_cols].applymap(lambda x: remove_spe_cha(x))
        self.df_json[other_str_cols] = self.df_json[other_str_cols].applymap(lambda x: remove_spe_cha(x))

        return 

    def convert(self, df_json, df_bq):
        """Compare logic is first to convert date, then will %, rest is others.
        """
        self.df_json = df_json
        self.df_bq = df_bq

        self.convert_date_cols()
        self.convert_percen_cols()
        self.convert_other_spe_cols()

        return self.df_json, self.df_bq
