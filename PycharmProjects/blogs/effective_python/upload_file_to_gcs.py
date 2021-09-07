from google.cloud import storage
import io
import os


bucket_name = "npd-65343-datalake-bd-9348-travels-npd-bd-ca-travel-raw"
key_path = r"C:\Users\guangqiiang.lu\Documents\lugq\workings\202009\migration\hands_on_ebi"
key_path = r"C:\Users\guangqiiang.lu\Documents\lugq\workings\202101\EBI_hands_on\SA_key"
key_file = "ex-9348-npd-nprdebi.json"
file_path = r"C:\Users\guangqiiang.lu\Downloads"
file_name = "sample.gz"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(key_path, key_file)


client = storage.Client()

client = storage.Client.from_service_account_json(os.path.join(key_path, key_file))
bucket = client.bucket(bucket_name)

blob = bucket.blob(file_name)
try:
    file_path = os.path.join(file_path, file_name)
    blob.upload_from_filename(file_path)
    print("File: {} has been uploaded into bucket:{}".format(file_name, bucket_name))
except Exception as e:
    print("Upload file with error:{}".format(e))


# os.system("gsutil cp {} gs://{}/{}".format(file_name, bucket_name, file_name))
