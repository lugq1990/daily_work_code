from google.cloud import translate
from google.cloud import storage
import logging
import unicodedata
import random
import os
import datetime
import pytz
import string
import re


def main(event, context):
    data = event
    project_id = os.environ.get('project_id')
    location = os.environ.get('location')
    destBucketName = os.environ.get('target_bucket')
    errorBucketName = os.environ.get('error_bucket')
    storageClient=storage.Client()
    translateClient=translate.TranslationServiceClient()
    parent = "projects/{}/locations/{}".format(project_id, location)
#    parent = translateClient.location_path(project_id, location)
    sourceBucket=storageClient.get_bucket(data['bucket'])
    errorBucket=storageClient.get_bucket(errorBucketName)
    blob=sourceBucket.blob(data['name'])
    logDict={
        "File": blob.name,
        "Function": "LANGUAGE IDENTIFICATION",
        "Status":"BEGINING"
        }
    logging.info(logDict)
    try:
        #phrase = getChars(blob)
        #detectedLanguageResponse = getLanguage(project_id,location,phrase)
        #languageCode=detectedLanguageResponse.languages[0].language_code
        text=getText(blob)
        languageCode = detectedLanguageAPI(text, translateClient, parent)
    except Exception as e:
        copy_to_error(sourceBucket,blob,errorBucket)
        logDict={
        "File": blob.name,
        "Function": "LANGUAGE IDENTIFICATION",
        "Exception": type(e),
        "Exception message": str(e),
        "Operation": "FILE HAS BEEN COPIED TO {}, IT SHOULD BE PROCESSED MANUALLY LATER".format(errorBucketName),
        "Status":"DONE"
        }
        logging.error(logDict)
        return None
    
    if languageCode[:2] == "en":
            blob.delete()
            logDict={
            "File": blob.name,
            "Function": "LANGUAGE IDENTIFICATION",
            "Mesagge": "SOURCE LANGUAGE OF THE FILE IS ENGLISH SO IT CAN´T BE TRANSLATED",
            "Operation": "DELETING THE FILE",
            "Status":"DONE"
            }
            logging.error(logDict)
            return None 

    copy_to_processing(storageClient,sourceBucket,blob,destBucketName,languageCode)
        
    
def getText(blob):
    srcContents = blob.download_as_string().decode('utf-8')
    return srcContents

def getLanguage(project_id,location,phrase):
    client=translate.TranslationServiceClient()
    parent = "projects/{}/locations/{}".format(project_id, location)
#    parent = client.location_path(project_id, location)
    srcLangResponse = client.detect_language(content=phrase,parent=parent)
    higherProbableLanguageCode=srcLangResponse.languages[0].language_code
    return srcLangResponse


def copy_to_processing(client,sourceBucket,blob,destBucketName,languageCode):
    appengine_timezone = pytz.timezone('America/Chicago')
    ts=str(datetime.datetime.now(appengine_timezone).isoformat(timespec='milliseconds')).replace(':','-')
    filepath= "/".join([languageCode,blob.name.split('/')[-1][:-4]+('_'+ts+'.txt')])
    sourceBucket.copy_blob(blob, client.get_bucket(destBucketName),filepath)
    logDict={
        "File": blob.name,
        "Function": "LANGUAGE IDENTIFICATION",
        "Message": "FILE WAS RENAMED AS {} AND COPIED TO {}".format(filepath,destBucketName),
        "Status":"DONE"
    }
    logging.info(logDict)
    blob.delete()

def copy_to_error(sourceBucket,blobObject,destBucket):
    sourceBucket.copy_blob(blobObject, destBucket,new_name=blobObject.name.split('/')[-1])
    blobObject.delete()


def _frac_words(text):
    len_ = len(text)
    if len_ > 0: 
        n_words=len([t for t in text if unicodedata.category(t)[0]=="L"])
        return float(n_words)/len_, len_
    else:
        return 0.0, 0


def _remove_number_symbol(sample_list):
    pattern = r'[^A-Za-z0-9À-ÖØ-öø-ÿ/]+'
    out_sample = []
    for sample in sample_list:
        if not sample.isascii():
            r = ""
            for x in sample:
                if not x.isdigit() and x not in string.punctuation:
                    r += x
            if r == '':
                continue
            out_sample.append(r)
        else:
            lat_res = ' '.join([x for x in [re.sub(pattern, '', sp) for sp in sample.split(' ')] if not x.isdigit() and x !=''])
            if lat_res == '':
                continue
            out_sample.append(lat_res)
    return out_sample
            


# Find a "good" random sample of paragraphs. We try to get large paragraphs with a high ratio of letters vs other characters
# (as they shoudl be more informative. The total number of characters to process will be CHAR_SIZE, so we can control the expense
#
def _find_samples(text):
    SAMPLE_SIZE=500
    CHAR_SIZE=4000
    WORD_CUTOFF= 0.6
    MIN_SAMPLE_SIZE = 5
    MIN_CHAR_SIZE = 4
    paragraphs=text.split("\n")
    # add remove number and other symbols here.
    paragraphs = _remove_number_symbol(paragraphs)
    paragraphs_sample= random.sample(paragraphs, k= min(SAMPLE_SIZE,len(paragraphs)))
    #Sort by size (max to min)
    list_paragraphs=sorted([(p,_frac_words(p)) for p in paragraphs_sample], key=lambda tple: -tple[1][1])
    #Filter out paragraphs with a word content too small
    list_paragraphs_good= [(par, fracs[1]) for par, fracs in list_paragraphs if fracs[0] > WORD_CUTOFF and fracs[1] > MIN_CHAR_SIZE]
    if len(list_paragraphs_good) ==0:
        #Fallback - We pick the largest paragraph and a random sample of the remaining paragraphs
        if len(paragraphs_sample) > 1:
            ks=[0] + random.sample(list(range(1,len(paragraphs_sample))), k=min(MIN_SAMPLE_SIZE,len(paragraphs_sample)-1))
        else:
            ks=[0]
        #This list wil lnever be empty if text is not empty
        list_paragraphs_good= [(list_paragraphs[k][0], list_paragraphs[k][1][1]) for k in ks if list_paragraphs[k][1][1] > 0]
    n_chars=0
    samples=[]
    #Append good paragraphs until we reach CHAR_SIZE total characters adding up lengths from all samples
    for par, size in list_paragraphs_good:
        if n_chars < CHAR_SIZE:
            if n_chars + size > CHAR_SIZE:
                samples.append(par[0:CHAR_SIZE - n_chars]) #Truncating the last parapgraph to have only CHAR_SIZE characters 
            else:
                samples.append(par)
            n_chars = n_chars + size #The text sent for translation will be of size 4000 or more                    
        elif len(samples) > 0:
            break
    return samples
#
# Uses _frac_words and _find_samples
#
def detectedLanguageAPI(text, client, parent):
    detectedLanguage = "und"
    if len(text) == 0:
        return detectedLanguage
    samples= _find_samples(text)
    #Call detect_language API for every sample paragraph
    responses=[]
    for sample in samples:
        response = client.detect_language(
                content=sample,
                parent=parent,
                mime_type='text/plain'  # mime types: text/plain, text/html
                )
        responses.append(response)
    #Collect the sum of confidences for every detected language and pick the largest one
    best_lang=dict()
    for response in responses:
        for language in response.languages:
            # The language detected
            lang_max_conf=language.language_code
        if lang_max_conf in best_lang.keys():
            best_lang[lang_max_conf]=best_lang[lang_max_conf] +language.confidence
        else:
            best_lang[lang_max_conf]=language.confidence
    #detected_language=max(list(best_lang.items()), key=lambda p: p[1])[0]
    
    list_new=[(lang, conf) for lang, conf in best_lang.items() if lang != "und"]
    if len(list_new) == 0:
        return "und"
    else:
        return max(list_new, key=lambda p: p[1])[0]



    
    
        
    


    