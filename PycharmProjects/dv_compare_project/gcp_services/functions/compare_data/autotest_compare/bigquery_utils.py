from google.cloud import bigquery

client = bigquery.Client()

def query_bigquery(query, schema):
    """ query BigQuery and return data with columns in schema

    although data type is specified in schema,
    returned DataFrame will have all data types being string

    Args:
        query(str): query for BigQuery
        schema(dict): dictionary of fields and their data type

    Returns:
        pandas.DataFrame: retrieved data from BigQuery, all columns being strings
    """
    data = client.query(query).to_dataframe(dtypes={field: str for field in schema})

    return data
