### This code should be used as a full load!
import time

from seal_api_util import SealAPIData, GCSUpload, BQLoadGCSFile


file_name = 'full_result_api.csv'


def retreve_data(event, context):
    """Get data from API Call based on Pubsub.

    Now logic is for one clause with one function.

    Args:
        event ([type]): [description]
        context ([type]): [description]
    """
    start_time = time.time()
    
    seal_api_data = SealAPIData()

    full_meta = seal_api_data.get_full_meta()
    if not full_meta:
        return
    
    end_time = time.time()
    
    # print("Final get {} contracts".format(len(full_meta.keys())))
    print("Full step takes: {} seconds.".format(round(end_time-start_time, 2)))

    # gcs_upload_obj = GCSUpload()
    # gcs_upload_obj.upload_data_into_gcs(full_meta)
    
    # gcs_file_path = 'gs://' + gcs_upload_obj.bucket_name + "/" + file_name
    
    # bq_load_gcs_file_obj = BQLoadGCSFile(gcs_file_path)
    # bq_load_gcs_file_obj.load_gcs_file_to_bq()


if __name__ == '__main__':
    retreve_data(1, 1)