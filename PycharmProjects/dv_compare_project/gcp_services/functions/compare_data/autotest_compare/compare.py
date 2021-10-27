import pandas as pd
def compare(expected, actual):
    """ compare two pandas.DataFrame

    Args:
        expected(pandas.DataFrame): expected result, usually from BigQuery or passed by tester
        actual(pandas.DataFrame): actual result, usually passed by Selenium

    Returns:
        status(boolean): indicator of whether expected matches actual
        missed_expected(pandas.DataFrame): rows in expected-actual
        missed_actual(pandas.DataFrame): rows in actual-expected
    """
    symmetric_difference = pd.concat([expected, actual]).drop_duplicates(keep=False)
    if symmetric_difference.empty:
        print("actual matches expected")
        status = True
        missed_expected = []
        missed_actual = []
    else:
        print("actual does not match expected")
        status = False
        missed_expected = expected.merge(actual, how='outer', indicator=True).loc[
            lambda x: x['_merge'] == 'left_only'].to_dict("records")
        missed_actual = expected.merge(actual, how='outer', indicator=True).loc[
            lambda x: x['_merge'] == 'right_only'].to_dict("records")
    return status, missed_expected, missed_actual