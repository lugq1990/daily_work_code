import time
import json
import os
import re
import json
import requests
import pandas as pd
import numpy as np
import datetime
import copy
from collections import defaultdict

from google.cloud import storage
from google.cloud import bigquery

import warnings

warnings.simplefilter("ignore")


# variables used 
DEFAULT_COLUMNS = ['contract_id', 'clause_name', 'clause_text', 'is_broad']
DEFAULT_TYPES = ['string', 'string', 'string', 'string']

bucket_name = "dv_test_bucket_lugq"
file_name = "full_result_api"
dataset_name = "auto_test"
table_name = "api_data"
each_batch = 100
clause_name = "NonSolicitationTerm"
base_url = "https://accenture-peii-01.seal-software.com/seal-ws/v5"
n_days_before = 100  # How many previous dates to be processed.


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
            "nonce":"{}".format(nonce_key),
        }

        auth_url = "{}/auths".format(self.base_url)

        auth_body = requests.post(auth_url, json=session_key_gen_body)

        auth_session_key = auth_body.headers.get('X-Session-Token')

        auth_header = {session_name: auth_session_key}

        return auth_header


class SealAPIData():
    def __init__(self, env='staging', n_days_before=n_days_before, clause_name="NonSolicitation") -> None:
        self.auth = _SealAuth(env)
        self.base_url = self.auth.base_url
        self.clause_name = clause_name
        self.n_days_before = n_days_before
        # Do only one time if object is instanted.
        self.auth_header = self.auth.get_auth_header()
  
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

    def get_full_meta(self, added_clause_names=None, total_contracts=None):
        """GET full metadata for that contract based on `each_batch` size to get loop with `n_batch` to process.
        
        added_clause_name: could be a list of terms!

        Returns:
            [full_result]: [Full metadata with type JSON: ({contract_id: [(clause_name, text), ...]})]
        """
        if not total_contracts:
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
        full_result = defaultdict(list)
        
        # convert full_result into list for saving data into table.
        # full_result = []

        for n_b in range(n_batch):
            if n_b % 20 == 0:
                print("This is {}th batch".format(n_b))
            
            url = "{}/contracts?offset={}&limit={}&expand=metadata&fq=+{}:*".format(self.base_url, n_b*each_batch, each_batch, self.clause_name)
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
                
                # let's change it into a {}
                single_sample_clauses = {}
                for j in range(len(single_sample)):
                    # loop for each clause
                    clause_name_each = single_sample[j]['name']
                    clause_texts = single_sample[j]['values']
                    
                    # NOTED: without check with startswith, but should be equal! added_clause_name could be a list of terms.
                    if added_clause_names and isinstance(added_clause_names, str):
                        added_clause_names = [added_clause_names]
                    
                    # logic here: if not equal with clause_name and provided a list of clause names but not in provided clause name, then just passs    
                    if added_clause_names and clause_name_each in added_clause_names:
                        clause_text_list = []
                        for t in range(len(clause_texts)):
                            clause_text = clause_texts[t]['value']
                            clause_text_list.append(clause_text)

                        clause_text_str = "~".join(clause_text_list)

                        single_sample_clauses[clause_name_each] = clause_text_str

                        # single_sample_clauses.append([clause_name_each, clause_text_str])
                    else:
                        continue
                
                # full_result[contract_id] = single_sample_clauses
                # full_result.extend(single_sample_clauses)
                full_result[contract_id] = single_sample_clauses

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
        
        print("Get {} daily contracts for daily filter".format(len(items)))
        
        for i in range(len(items)):
            if 'id' not in items[i]:
                continue
            ids_list.append(items[i]['id'])
        
        return ids_list
    
    def get_daily_filter_metadata(self, added_clause_names=None):
        """Get daily metadata with [contract_id, clause_name, cluase_text] for each contract.

        Returns:
            [type]: [description]
        """
        # todo: Should add a function to get contract_ids from BQ and get difference.
        query_ids_list = self._get_daily_contract_ids()
        
        daily_contracts_metadata = []
        
        for id in query_ids_list:
            url = "{}/contracts/{}/metadata?offset=0&limit=99999".format(self.base_url, id)
            
            response_json = self.get_api_response_json(url)
            
            # Based on each contracts to get each clause name and metadata.
            items = response_json.get('items')
            
            # todo: there should be changed for daily for added clause name list
            for i in range(len(items)):
                each_clause = items[i]
                each_clause_name = each_clause['name']
                
                # NOTED: without check with startswith, but should be equal! added_clause_name could be a list of terms.
                if added_clause_names and isinstance(added_clause_names, str):
                    added_clause_names = [added_clause_names]

                # TODO: change for daily!
                # logic here: if not equal with clause_name and provided a list of clause names but not in provided clause name, then just passs    
                if added_clause_names and clause_name_each in added_clause_names:
                    clause_text_list = []
                    for t in range(len(clause_texts)):
                        clause_text = clause_texts[t]['value']
                        clause_text_list.append(clause_text)

                    clause_text_str = "~".join(clause_text_list)

                    single_sample_clauses.append([contract_id, clause_name_each, clause_text_str])
                else:
                    continue
                    
                clause_text_list =[]
                # TODO: at least for now just to store each clause's value into a list, but in fact shoud be changed here to be a string
                for j in range(len(each_clause['values'])):
                    clause_text_list.append(each_clause['values'][j]['value'])
                    
                # Get each contract_id with metadata
                daily_contracts_metadata.append([id, each_clause_name, clause_text_list])
                
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
            
    def transform(self, contracts, return_df=True):
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
        
        print("There are {} contracts to be processed.".format(len(contracts)))
        
        # Highlight: business logic filter here!
        for contract_id in contracts.keys():
            is_broad, fail_reason = self._compare_term(contracts.get(contract_id))
            contracts.get(contract_id)["is_broad"] = is_broad
            contracts.get(contract_id)['broad_reason'] = fail_reason

        if return_df:
            # convert into a DataFrame type
            contracts = pd.DataFrame(contracts).T.reset_index().rename(columns={"index":"contract_id"})

        return contracts

    def _compare_term(self, metadata_dic, text_term="NonSolicitation", date_term='Normalized_NonSolicitationTerm', user_define_month_thre=18):
        """
        _compare_term Main compare function with implement for business logic.

        Args:
            metadata_dic ([type]): [description]
            text_term (str, optional): [description]. Defaults to "NonSolicitation".
            date_term (str, optional): [description]. Defaults to 'Normalized_NonSolicitationTerm'.
            user_define_month_thre (int, optional): [description]. Defaults to 18.

        Raises:
            ValueError: [description]
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        is_broad = False
        fail_reason_dict = {}
        fail_reason_dict['keyword'] = "Nonsolicitation contain keyword: {}"
        fail_reason_dict['date'] = "NonsolicitationTerm over 18 months: {}"
        fail_reason_dict['not_found_text'] = "Couldn't get Nonsolicitation text"
        fail_reason_dict['not_found_date'] = "Couldn't get Nonsolicitation Term"

        fail_reason = ""

        clause_text = metadata_dic.get(text_term)
        if not clause_text:
            # not text, just is borad
            fail_reason = fail_reason_dict.get("not_found_text")
            raise ValueError(fail_reason)
        
        for kw in self.keywords:
            if kw in clause_text:
                # get key-words in clause_text
                is_broad = True
                fail_reason = fail_reason_dict.get("keyword").format(kw)
                break

        if is_broad:
            # no need to check date
            return is_broad, fail_reason

        clause_date = metadata_dic.get(date_term)
        if not clause_date:
            is_broad = True
        else:
            # get normalized term, if not with normalized type, then just pass as true, use `re` to get satisfied type
            groups = re.match(r"(\d+) (\w+)", clause_date).groups()
            if len(groups) != 2:
                # print("Not good format with {}: get:{}".format(date_term, clause_date))
                is_broad = True
                fail_reason = "Not good format with {}: get:{}".format(date_term, clause_date)
                # raise ValueError("Not good format with {}: get:{}".format(date_term, clause_date))
            else:
                months, month_type = int(groups[0]), groups[1]
                if month_type != 'months':
                    print("Get date type: {}, should be: months!".format(month_type))
                    fail_reason = "Get date type: {}, should be: months!".format(clause_date)
                    return True, fail_reason
                else:
                    if months <= user_define_month_thre:
                        is_broad = False
                    else:
                        is_broad = True
                        fail_reason = fail_reason_dict.get("date").format(str(clause_date))
        return is_broad, fail_reason
    
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
            data = [x.replace('\n', '').strip() for x in data]
            
        return data
                
                