"""Daily load work to retrieve data with API to store data in BQ.

Detail steps:
1. Get full contract_id in BQ as historical base.
2. Get `n_days` historical contract_id to be compared.
3. Get different contract_id from SEAL AND BQ 
4. Get metadata of different contract_id
5. Dump new contract_id metadata into GCS
6. Load GCS files and do transformation
7. Dump result into GCS files.
8. Load GCS's file into BQ
"""
import time

from seal_api_util import SealAPIData, GCSUpload, BQLoadGCSFile


file_name = "daily_result_api.csv"


def retreve_data(event, context):
    """Get data from API Call based on Pubsub.

    Now logic is for one clause with one function.

    Args:
        event ([type]): [description]
        context ([type]): [description]
    """
    start_time = time.time()
    
    seal_api_data = SealAPIData()

    daily_meta = seal_api_data.get_daily_metadata_with_date_filter()
    
    end_time = time.time()
    
    # print("Final get {} contracts".format(len(full_meta.keys())))
    print("Full step takes: {} seconds.".format(round(end_time-start_time, 2)))
    
    if not daily_meta:
        return

    # gcs_upload_obj = GCSUpload(file_name=file_name)
    # gcs_upload_obj.upload_data_into_gcs(daily_meta)
    
    # gcs_file_path = "gs://" + gcs_upload_obj.bucket_name + '/' + file_name
    
    # bq_load_gcs_file_obj = BQLoadGCSFile(gcs_file_path=gcs_file_path, mode='append')
    # bq_load_gcs_file_obj.load_gcs_file_to_bq()


if __name__ == '__main__':
    retreve_data(1, 1)