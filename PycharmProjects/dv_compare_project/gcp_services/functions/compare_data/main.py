from autotest_compare.pubsub_utils import publish, parse_test, construct_message
from autotest_compare.parser import parse_data
from autotest_compare.bigquery_utils import query_bigquery
from autotest_compare.compare import compare

def run_test_case(event, context):
    """Triggered from a message on a Cloud Pub/Sub topic.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """

    # receive test case
    identity, query, data, schema, mode = parse_test(event, context)

    if mode == "compare":
        # parse data
        actual = parse_data(data, schema)
        expected = parse_data(query_bigquery(query, schema), schema)

        # compare
        print("comparing")
        status, missed_expected, missed_actual = compare(expected, actual)

        # construct message for test result and publish
        message = construct_message(identity, status,
                                    missed_expected=missed_expected,
                                    missed_actual=missed_actual)
        publish(message)

    else:
        print("mode is not compare")
        

    
