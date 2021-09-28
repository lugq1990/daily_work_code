import datetime
from googleapiclient import discovery

# Try to add with random string for job_id is same problem.
import random
import string

random_string = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))


id = "sbx-65343-autotest13--199c6387"
project_id = "projects/{}".format(id)
bucket_name = "schedule_with_platform"
job_name = "training_" + datetime.datetime.now().strftime('%y%m%d_%H%M%S') +"_"+ random_string

def submit_job(event, context):
    print("Start to trigger ai jobs")
    training_inputs = {
        'scaleTier': 'BASIC',
        'packageUris': [f"gs://{bucket_name}/iris_trainer-0.1.tar.gz"],
        'pythonModule': 'iris_trainer.training',
        'region': 'us-east1',
        'jobDir': f"gs://{bucket_name}",
        'runtimeVersion': '2.2',
        'pythonVersion': '3.7',
    }
    job_spec = {"jobId": job_name, "trainingInput": training_inputs}

    cloud_ml = discovery.build('ml', 'v1', cache_discovery=False)
    request = cloud_ml.projects().jobs().create(body=job_spec, parent=project_id)
    response = request.execute()
    print(response)
    try:
        print('get response code: ',response.state)
    except:
        print("response doesn't support status_code")
        print("response has attr: ", dir(response))
        