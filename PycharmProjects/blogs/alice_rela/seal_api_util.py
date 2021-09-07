"""Util functions changes for SEAL API related.
"""
import time
import json
import os
import json
import requests
import pandas as pd
import numpy as np
import datetime
import copy

from google.cloud import storage
from google.cloud import bigquery

import warnings

from six import b

warnings.simplefilter("ignore")


# variables used 
DEFAULT_COLUMNS = ['contract_id', 'clause_name', 'clause_text', 'is_broad']
DEFAULT_TYPES = ['string', 'string', 'string', 'string']

bucket_name = "dv_test_bucket_lugq"
file_name = "full_result_api"
dataset_name = "auto_test"
table_name = "api_data"
each_batch = 100
clause_name = "NonSolicitation"
base_url = "https://accenture-peii-01.seal-software.com/seal-ws/v5"
n_days_before = 10  # How many previous dates to be processed.


class _SealAuth:
    def __init__(self, env='staging') -> None:
        self.env = env
        if env == 'staging':
            self.base_url = "https://accenture-peii-01.seal-software.com/seal-ws/v5"
            self.principal = "A30899DIRPSealAPI"
            self.password = "lzy5*SaP*BjtNL+1%)C="
        elif env == 'production':
            self.base_url = "https://accenture-peii-03.seal-software.com/seal-ws/v5"
            self.principal = "A30899DIRPSealAPI"
            self.password = "lzy5*SaP*BjtNL+1%)C="
        
    def get_auth_header(self):
        """Get auth key based on URL.

        Returns:
            [auth_header]: [Auth header for seal]
        """
        session_name = "X-Session-Token"

        # automate session key generated
        nonce_key = requests.get("{}/security/nonce".format(self.base_url)).text

        # get x-session-key
        session_key_gen_body = {
            "principal": self.principal,
            "password": self.password,
            "nonce":"{}".format(nonce_key)
        }

        auth_url = "{}/auths".format(self.base_url)

        auth_body = requests.post(auth_url, json=session_key_gen_body)

        auth_session_key = auth_body.headers.get('X-Session-Token')

        auth_header = {session_name: auth_session_key}

        return auth_header


class SealAPIData():
    def __init__(self, env='staging', n_days_before=n_days_before) -> None:
        self.auth = _SealAuth(env)
        self.base_url = self.auth.base_url
        self.clause_name = clause_name
        self.n_days_before = n_days_before
        # Do only one time if object is instanted.
        self.auth_header = self.auth.get_auth_header()
        self.transform = UDFTransform()
  
    @staticmethod
    def get_response_status(response):
        """Ensure get right response.

        Args:
            response ([type]): [description]

        Returns:
            [status]: [True if get 200 response.]
        """
        status = True
        if response.status_code != 200:
            print("Get Error to call API with status_code: {} with error: `{}`".format(response.status_code, response.text))
            status = False

        return status
    
    def get_api_response_json(self, url):
        """Get response JSON data based on url.

        # todo: there will be error handler here.
        Args:
            url ([type]): [description]

        Returns:
            [type]: [description]
        """
        
        response = requests.get(url, headers=self.auth_header)

        status = self.get_response_status(response)
        if not status:
            print("Bad response to get full contract count!")
            return

        response_json = response.json()
        
        return response_json
        
    def get_contract_num(self):
        """Get how many contracts to be processed.

        Returns:
            [total_contracts]: [How many contracts for this clause]
        """
        # first to get how many files.
        # After remove key word: `&expand=metadata` faster than before!
        url = "{}/contracts?offset=0&limit=1&fq=+{}:*".format(self.base_url, self.clause_name)
        
        res = self.get_api_response_json(url)

        total_contracts = res['meta']['totalCount']
        print("There are {} satisfied contracts to be processed.".format(total_contracts))

        return total_contracts

    def get_full_meta(self):
        """GET full metadata for that contract based on `each_batch` size to get loop with `n_batch` to process.

        Returns:
            [full_result]: [Full metadata with type JSON: ({contract_id: [(clause_name, text), ...]})]
        """
        total_contracts = self.get_contract_num()

        if not total_contracts:
            print("Nothing get with API for full meta")
            return

        # How many batches to be processed based on each batch size    
        n_batch = int(np.ceil(total_contracts / each_batch))
        n_samples = each_batch
        print("Based on each batch with {} samples, needed {} batches".format(each_batch, n_batch))

        # get full conctracts with each clause and contents
        # change from list into JSON: {contract_id: [(clause_name, text), ...]}
        full_result = {}
        
        # convert full_result into list for saving data into table.
        full_result = []

        for n_b in range(n_batch):
            if n_b % 20 == 0:
                print("This is {}th batch".format(n_b))
            
            url = "{}/contracts?offset={}&limit={}&expand=metadata&fq=+{}:*".format(base_url, n_b*each_batch, each_batch, clause_name)
            # solve Python Requests throwing SSLError error.
            # This should be sovled in real production.
            # todo: Maybe with the same auth_header will face not-validated auth once timeout for auth, to be checked.
            response = requests.get(url, headers=self.auth_header, verify=False)
            status = self.get_response_status(response)
            if not status:
                print("Bad response for batch call!")
                return
            
            res = response.json()

            if n_b == n_batch - 1:
                # This is last batch
                n_samples = total_contracts - each_batch * (n_batch - 1)

            for i in range(n_samples):
                # loop API output as each contract.
                single_sample = res['items'][i]['metadata']['items']
                contract_id = res['items'][i]['id']
                
                single_sample_clauses = []
                for j in range(len(single_sample)):
                    # loop for each clause
                    clause_name_each = single_sample[j]['name']
                    clause_texts = single_sample[j]['values']

                    if not clause_name_each.startswith(clause_name):
                        continue
                    
                    clause_text_list = []
                    for t in range(len(clause_texts)):
                        clause_text = clause_texts[t]['value']
                        clause_text_list.append(clause_text)
                        
                    clause_text_str = "~".join(clause_text_list)
                        
                    single_sample_clauses.append([contract_id, clause_name_each, clause_text_str])
                
                # full_result[contract_id] = single_sample_clauses
                full_result.extend(single_sample_clauses)

        # Add transformation here.
        full_result = self.transform.transform(full_result)
        
        return full_result
    
    def _get_filter_query_date(self):
        """To get a filter date string for API based on `n_days_before` from today's first start.

        Returns:
            [query_date]: [API format string like: [2021-08-12T00:00:00Z TO 2021-08-13T00:00:00Z]]
        """
        filter_time_format = "%Y-%m-%dT%H:%M:%SZ"

        now = datetime.datetime.now()

        start_of_day = datetime.datetime(now.year,now.month,now.day)
        yes_date = start_of_day - datetime.timedelta(days=self.n_days_before)

        now_format = start_of_day.strftime(filter_time_format)
        yes_format = yes_date.strftime(filter_time_format)

        query_date = "[{} TO {}]".format(yes_format, now_format)
        
        return query_date
    
    def _get_daily_contract_ids(self):
        query_filter_date = self._get_filter_query_date()
        
        url = "{}/contracts?offset=0&limit=100000&fq=+{}:*%+StartDate:{}".format(self.base_url, self.clause_name, query_filter_date)
        
        response_json = self.get_api_response_json(url)
        
        ids_list = []
        items = response_json['items']
        
        print("GET {} contract_ids for {} days filter.".format(len(items), self.n_days_before))
        
        for i in range(len(items)):
            if 'id' not in items[i]:
                # in case we don't get each contract_id
                continue
            ids_list.append(items[i]['id'])
        
        return ids_list
    
    def get_daily_metadata_with_date_filter(self):
        """Get daily metadata with [contract_id, clause_name, cluase_text] for each contract.

        Returns:
            [type]: [description]
        """
        # todo: Should add a function to get contract_ids from BQ and get difference.
        query_ids_list = self._get_daily_contract_ids()
        
        daily_contracts_metadata = []
        
        for id in query_ids_list:
            url = "{}/contracts/{}/metadata?offset=0&limit=99999".format(self.base_url, id)
            
            index_num =  query_ids_list.index(id) 
            if index_num % 10 ==0:
                print("Already get {} contracts metadata.".format(index_num))
            
            response_json = self.get_api_response_json(url)
            
            # Based on each contracts to get each clause name and metadata.
            items = response_json.get('items')
            
            for i in range(len(items)):
                each_clause = items[i]
                each_clause_name = each_clause['name']
                if not each_clause_name.startswith(clause_name):
                    # if this contract_id doestn' contain with clause_name, then just pass won't store this ID.
                    continue
                
                clause_text_list =[]
                # TODO: at least for now just to store each clause's value into a list, but in fact shoud be changed here to be a string
                for j in range(len(each_clause['values'])):
                    clause_text_list.append(each_clause['values'][j]['value'])
                    
                # Get each contract_id with metadata
                daily_contracts_metadata.append((id, each_clause_name, clause_text_list))
                
        print("Get {} daily contract metadata.".format(len(daily_contracts_metadata)))
                
        # Add transformation here.
        daily_contracts_metadata = self.transform.transform(daily_contracts_metadata)
        
        return daily_contracts_metadata
    
    
class UDFTransform:
    def __init__(self, keyword_file_name='keywords.txt', date_term_file_name='date_term.txt') -> None:
        """Transform a list of contracts metadata to check is broad or not.

        Args:
            keyword_file_name (str, optional): Which keyword file to be loaded. Defaults to 'keywords.txt'.

        Raises:
            FileNotFoundError: keyword_file_name doesn't exist.
        """
        self.keywords = self._get_file_content(keyword_file_name)
        self.date_term = self._get_file_content(date_term_file_name)
            
    def transform(self, contracts_list):
        """Transform a list of data based on last index value as a list of clause text, add with `is_broad` as True or False at last.

        Args:
            contracts_list (list): List of clauses: [[contract_id, clause_name, [clause_text1, clause_text2, ..]]]

        Raises:
            ValueError: keyword not exist
            ValueError: should be list of clause data

        Returns:
            [list]: [transformed data with each clause appended with True or False]
        """
        if not self.keywords:
            raise ValueError("keyword object not exist, please check.")
        
        if not isinstance(contracts_list, list):
            raise ValueError("Not is only support with List object to be transformed.")
        
        print("There are {} contracts to be processed.".format(len(contracts_list)))
        
        contracts_list_cp = copy.deepcopy(contracts_list)
        
        for single_contract in contracts_list_cp:
            # todo: there should be a better way to do this.
            is_broad = False
            for i in range(len(single_contract[-1])):
                for kw in self.keywords:
                    for date in self.date_term:
                        if kw in single_contract[-1][i] and date not in single_contract[-1][i]:
                            is_broad = True
                            break
                        
            single_contract.append(is_broad)

            contracts_list_cp[i] =  single_contract
        
        return contracts_list_cp
    
    @staticmethod   
    def _get_file_content(file_name):
        """Read file content line by line.

        Args:
            file_name (str): file to be loaded

        Raises:
            FileNotFoundError: File doesn't exist

        Returns:
            list: a list of file content
        """
        if not os.path.exists(file_name):
            raise FileNotFoundError("File: {} not exist.".format(file_name))
        
        with open(file_name, 'r') as f:
            data = f.readlines()
            data = [x.replace('\n', '') for x in data]
            
        return data
            

class GCSUpload():
    def __init__(self, bucket_name=bucket_name, file_name=file_name) -> None:
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(bucket_name)
        self.file_name = file_name
        
    def upload_data_into_gcs(self, obj):
        """Upload object into GCS based on different type of data.

        Args:
            obj ([type]): [description]
        """
        if isinstance(obj, list):
            self.file_name += '.csv'
            self.upload_list_to_gcs_with_csv(obj)
        elif isinstance(obj, dict):
            self.file_name += '.json'
            self.upload_json_to_gcs(obj)

    def upload_json_to_gcs(self, json_obj, check_exist=True):
        """Upload final JSON object to GCS directly.

        Args:
            json_obj ([JSON]): [JSON object to be processed]
            check_exist (bool, optional): [Whether or not to check file in GCS?]. 
                Defaults to True.
        """ 
        blob = self.bucket.blob(self.file_name)
        #blob.upload_from_filename(file_name)
        blob.upload_from_string(json.dumps(json_obj), 'text/json', timeout=600)

        # get bucket files 
        if check_exist:
            self.check_file_exist_or_not(file_name)

    def upload_list_to_gcs_with_csv(self, list_obj, 
                    columns=['contract_id', 'clause_name', 'clause_text'], 
                    check_exist=True, check_broad_txt_list=['acknowledges']):
        """Upload a list object into GCS with CSV file type.

        Args:
            list_obj ([type]): [description]
            columns (list, optional): [description]. Defaults to ['contract_id', 'clause_name', 'clause_text'].
            check_exist (bool, optional): [description]. Defaults to True.
            check_broad_txt_list (list, optional): [description]. Defaults to ['acknowledges'].

        Returns:
            [type]: [description]
        """
        
        df = pd.DataFrame(list_obj, columns=columns)
        
        # # This should be changed for checking
        # todo: This should added with a function
        # def _is_broad(x):
        #     x_split = x.split(" ")
        #     for d in x_split:
        #         if d in check_broad_txt_list:
        #             return True
        #     return False
        
        # df['is_broad'] = df['clause_text'].apply(lambda x: _is_broad(x))

        blob = self.bucket.blob(self.file_name)
        #blob.upload_from_filename(file_name)
        blob.upload_from_string(df.to_csv(index=False), 'text/csv', timeout=600)

        # get bucket files 
        if check_exist:
            self.check_file_exist_or_not(file_name)
            
    def check_file_exist_or_not(self, file_name):
        files = [x.name for x in self.bucket.list_blobs()]
        exist = file_name in files
        print("File: {} exist or not: {}".format(file_name, str(exist)))
        
        return exist
        
        
class BQLoadGCSFile():
    def __init__(self, gcs_file_path) -> None:
        self.gcs_file_path = gcs_file_path
            
    # Use BQ client to upload files into BQ
    def load_gcs_file_to_bq(self, column_names=DEFAULT_COLUMNS, 
                            column_types=DEFAULT_TYPES, 
                            extension='csv',
                            mode='truncate'):
        """Load GCS file into BQ.

        Support with different type of file.

        Args:
            column_names ([type], optional): [description]. Defaults to DEFAULT_COLUMNS.
            column_types ([type], optional): [description]. Defaults to DEFAULT_TYPES.
        """
        client = bigquery.Client()

        dataset = client.get_dataset(dataset_name)
        table = dataset.table(table_name)

        schema_list = []
        for c, t in zip(column_names, column_types):
            schema_list.append(bigquery.SchemaField(c, t))
        
        if mode == 'truncate':
            load_mode = bigquery.WriteDisposition.WRITE_TRUNCATE
        elif mode == 'append':
            load_mode = bigquery.WriteDisposition.WRITE_APPEND

        job_config = bigquery.LoadJobConfig(
            schema=schema_list,
            skip_leading_rows=1,
            # The source format defaults to CSV, so the line below is optional.
            source_format=bigquery.SourceFormat.CSV,
            write_disposition=load_mode
        )

        job = client.load_table_from_uri(self.gcs_file_path, table, job_config=job_config)
        # Only we finished.
        job.result()
        