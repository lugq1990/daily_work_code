# -*- coding:utf-8 -*-
"""
this is with system log functionality

@author: Guangqiang.lu
"""
import redis
import time
import logging
import datetime
import bisect


QUIT = False


conn = redis.Redis()


SEVERITY = {
    logging.DEBUG: 'debug',
    logging.INFO: 'info',
    logging.WARNING: 'warning',
    logging.ERROR: 'error',
    logging.CRITICAL: 'critital',
}

SEVERITY.update((name, name) for name in SEVERITY)

def log_recent(conn, name, message, servrity=logging.INFO, pipe=None):
    servrity = str(SEVERITY.get(servrity, servrity)).lower()
    destination = 'recent:%s:%s' % (name, servrity)

    message = time.asctime() + " " + message

    pipe = pipe or conn.pipeline()
    pipe.lpush(destination, message)
    pipe.ltrim(destination, 0, 99)
    pipe.execute()


def log_common(conn, name, message, severity=logging.INFO, timeout=5):
    severity = str(SEVERITY.get(severity, severity)).lower()
    destination = "common:%s:%s" % (name, severity)
    start_key = destination + ':start'
    pipe = conn.pipeline()
    end = time.time() + timeout
    while time.time() < end:
        try:
            pipe.watch(start_key)
            now = datetime.datetime.utcnow().timetuple()
            hour_start = datetime.datetime(*now[:4]).isoformat()

            exiting = pipe.get(start_key)
            pipe.multi()
            if exiting and exiting < hour_start:
                pipe.rename(destination, destination + ":last")
                pipe.rename(start_key, destination+":pstart")
                pipe.set(start_key, hour_start)
            pipe.zincrby(destination, message)
            log_recent(pipe, name, message, severity, pipe)
            return
        except redis.exceptions.WatchError:
            continue


PRICISION = [1, 5, 60, 300, 3600, 18000]

def update_counter(conn, name, count=1, now=None):
    now = now or time.time()
    pipe = conn.pipeline()
    for prec in PRICISION:
        pnow = int(now / prec) * prec
        hash = "%s:%s" % (prec, name)
        pipe.zadd('known:', hash, 0)
        pipe.hincrby("count:" + hash, pnow, count)
    pipe.execute()


def get_counter(conn, name, precision):
    hash = "%s:%s" % (precision, name)
    data = conn.hgetall("count:" + hash)
    to_return = []
    for k, v in data.iteritems():
        to_return.append((int(k), int(v)))

    to_return.sort()
    return to_return


def clean_counters(conn):
    pipe = conn.pipeline(True)
    passes = 0
    while not QUIT:
        start = time.time()
        index = 0
        while index < conn.zcard("know:"):
            hash = conn.zrange("know:", index, index)
            index += 1
            if not hash:
                break

            hash = hash[0]
            prec = int(hash.partition(":")[0])
            bprec = int(prec // 60) or 1
            if passes % bprec:
                continue

            hkey = "count:" + hash
            cutoff = time.time()
            samples = map(int, conn.keys(hkey))
            samples.sort()
            remove = bisect.bisect_right(samples, cutoff)

            if remove:
                conn.hdel(hkey, *samples[:remove])
                if remove == len(samples):
                    try:
                        pipe.watch(hkey)
                        if not pipe.hlen(hkey):
                            pipe.multi()
                            pipe.zrem("know:" , hash)
                            pipe.execute()
                            index -= 1
                        else:
                            pipe.unwatch()
                    except redis.exceptions.WatchError:
                        pass

        passes += 1
        duration = min(int(time.time() - start) + 1, 60)
        time.sleep(max(60 - duration, 1))

