"""
Use Bigquery SQL syntax to get only difference between two SQLs.

Sample:
(select documentid from `sbx-65343-autotest13--199c6387.alice_insights_30899.dashboard_metadata_acc_vw` except distinct select distinct(documentid) from `sbx-65343-autotest13--199c6387.alice_insights_30899.dashboard_metadata_acc_vw`) 
union all 
(select distinct(documentid) from `sbx-65343-autotest13--199c6387.alice_insights_30899.dashboard_metadata_acc_vw` except distinct select documentid from `sbx-65343-autotest13--199c6387.alice_insights_30899.dashboard_metadata_acc_vw`)
"""

from google.cloud import bigquery


class DataCompareDESQLBase:
    def __init__(self) -> None:
        self.client = bigquery.Client()

    def compare(self, sql1, sql2, schema_compare=True, **kwargs):
        """
        compare Compare two SQLs output without real data row compare.

        Logic here:
            1. Get count of these 2 SQLs, if row number is not equal, False
            2. Get column number of 2 SQLs, if not equal then False
            3. Get difference between 2 SQLs, if there is any data then False

        Args:
            sql1 (str): First SQL
            sql2 (str): Second SQL
        """
        # 1. compare count number.
        res1 = self.client.query(sql1).result()
        res2 = self.client.query(sql2).result()

        count_res_1 = res1.total_rows
        count_res_2 = res2.total_rows

        if count_res_1 != count_res_2:
            return False

        # 2. compare schema number.
        # todo: there should be a schema name and type compare if needed.
        if schema_compare:
            schema_compare_res = self._compare_schema(res1, res2)

        if not schema_compare_res:
            print("Schema compare fail.")
            return schema_compare_res

        # 3. build compare SQL to comapre diff data number
        compare_merge_sql = (
            "({} except distinct {}) union all ({} except distinct {})".format(
                sql1, sql2, sql2, sql1
            )
        )
        merge_res = self.client.query(compare_merge_sql).result()

        merge_total_num = merge_res.total_rows
        if merge_total_num > 0:
            return False

        return True

    @staticmethod
    def _compare_schema(row_iter_1, row_iter_2, compare_name=False, compare_type=True):
        """
        _compare_schema Compare both schema length, name and type.

        Args:
            row_iter_1 ([type]): [description]
            row_iter_2 ([type]): [description]
        """
        schema_1 = row_iter_1.schema
        schema_2 = row_iter_2.schema

        if len(schema_1) != len(schema_2):
            print(
                "Schema length not equal, first: {} second: {}".format(
                    len(schema_1), len(schema_2)
                )
            )
            return False

        def _get_diff_lists(lst1, lst2):
            common_set = set(lst1) & set(lst2)

            diff_list_1 = list(set(lst1) - common_set)
            diff_list_2 = list(set(lst2) - common_set)

            return diff_list_1, diff_list_2

        if compare_name:
            schema_name_list_1 = [s.name for s in schema_1]
            schema_name_list_2 = [s.name for s in schema_2]

            diff_schema_1, diff_schema_2 = _get_diff_lists(
                schema_name_list_1, schema_name_list_2
            )

            if diff_schema_1 or diff_schema_2:
                return False

        if compare_type:
            schema_type_list_1 = [s.field_type for s in schema_1]
            schema_type_list_2 = [s.field_type for s in schema_2]
            diff_type_1, diff_type_2 = _get_diff_lists(
                schema_type_list_1, schema_type_list_2
            )

            if diff_type_1 or diff_type_2:
                return False

        return True


if __name__ == "__main__":
    sql_1 = "select clientcountry, documentid from `sbx-65343-autotest13--199c6387.alice_insights_30899.dashboard_metadata_acc_vw`"
    sql_2 = "select clientcountry, documentid as t from `sbx-65343-autotest13--199c6387.alice_insights_30899.dashboard_metadata_acc_vw` group by documentid, clientcountry"
    tmp_sql = "select count(1) as t from `sbx-65343-autotest13--199c6387.alice_insights_30899.dashboard_metadata_acc_vw`"

    data_compare_obj = DataCompareDESQLBase()
    res = data_compare_obj.compare(sql_1, sql_2)
    print(res)
