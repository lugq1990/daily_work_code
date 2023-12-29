import boto3



# s3_client = boto3.client("s3", aws_access_key_id=AWSAccessKeyId, aws_secret_access_key=AWSSecretKey, verify=False)
# s3 = boto3.resource("s3")
# bucket = s3.Bucket("lugq-2019-test")
s3 = boto3.resource("s3", aws_access_key_id=AWSAccessKeyId, aws_secret_access_key=AWSSecretKey, verify=False)
file_name = "mml_to_sftp.sh"
bucket_name = "lugq-2019-test"
client = s3.meta.client
client.verify = False
client.upload_file(file_name, bucket_name, file_name)


s3 = boto3.Session(aws_secret_access_key=AWSSecretKey, aws_access_key_id=AWSAccessKeyId, verify=False).resource('s3')

bucket_name = "lugq-2019-test"
my_bucket = s3.Bucket(bucket_name)

file_name = "mml_to_sftp.sh"
bucket_name = "lugq-2019-test"

bucket = s3.Bucket("lugq-2019-test")

print([x for x in my_bucket.objects.all()])

content = open(file_name, 'rb')

s3.put_object(Bucket=bucket_name, Key=file_name, Body=content)

try:
    s3.upload_file(file_name, bucket_name, file_name)
except Exception as e:
    print("Upload file with error:",  e)

s3.meta.client.Bucket(bucket_name).put_object(Key=file_name)