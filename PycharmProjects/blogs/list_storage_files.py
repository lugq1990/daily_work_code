from google.cloud import storage
import pandas as pd

client = storage.Client()

bucket_name = "npd-65343-datalake-bd-11811-dsd-npd-bd-ca-dsdgcs-raw"

bucket = client.get_bucket(bucket_name)

blobs = list(bucket.list_blobs())

file_names = [x.name for x in blobs]


df = pd.DataFrame(file_names, columns=['file_name'])


file_name = "ContactRoutingDetail_2020-11-13-063450.csv"
blob = bucket.get_blob(file_name)


# Used to see the source code of each machine learning algorithm.

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


from pyspark.ml.classification import RandomForestClassifier


from xgboost import XGBClassifier

xgb = XGBClassifier()

from sklearn.ensemble import AdaBoostClassifier

abc = AdaBoostClassifier()


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import train_test_split
# Based on libsvm for real training.
from sklearn.svm import SVC

from docx import Document

