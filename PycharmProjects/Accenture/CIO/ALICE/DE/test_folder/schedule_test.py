# -*- coding:utf-8 -*-
"""This is to test the python schedule module to run a callable function or something else
to run the job every wanted time """

import time
import schedule
import datetime
import functools

# here I could just add some log for the later step
def with_logging(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("Now start to execute the job.")
        result = func(*args, **kwargs)
        print("Job finished!")
        return result
    return wrapper

@with_logging
def work():
    print("I'm starting working ... at %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

@with_logging
def second():
    print("During the work for second function with time : %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

# This function is used to just to execute the job just once.
@with_logging
def job_with_once():
    print("This job is just run with once!")
    return schedule.CancelJob

# This function is to be executed within between 5 to 10 seconds
def random_job():
    print("I'm starting working ... at %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


# here is for job parallel by using the python thread module
import threading
# first is to job to be executed
def job():
    print("I'm starting working ... at %s" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

# this is used to run the thread for the function
def run_schedule(job_func):
    job_thread = threading.Thread(target=job_func)
    job_thread.start()

n_threads = 8
for _ in range(n_threads):
    schedule.every(10).seconds.do(run_schedule, job)


# schedule.every().minutes.do(work)
# schedule.every(10).seconds.do(second)
# schedule.every().seconds.do(job_with_once())
# schedule.every(5).to(10).seconds.do(random_job)

while True:
    schedule.run_pending()
    time.sleep(1)