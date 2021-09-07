# -*- coding:utf-8 -*-
"""
This is to build with a website

@author: Guangqiang.lu
"""
import time


def check_token(conn, token):
    """store the token into a hash set, so that we could just get with time o(1)"""
    return conn.hget("token:", token)


def update_token(conn, token, user, item=None):
    """update tokens when user stay in the website"""
    timestamp = time.time()
    conn.hset("login:", token, user)
    conn.zadd("recent:", token, timestamp)

    if item:
        conn.zadd("viewed:"+ token, item, timestamp)
        conn.zremrangebyrank("viewed:" + token, 0, -26)


QUIT = False
LIMIT = 1000000

def clean_sessions(conn):
    while not QUIT:
        size = conn.zcard("recent:")
        if size <= LIMIT:
            time.sleep(1)
            continue

        end_index = min(size - LIMIT, 100)
        tokens = conn.zrange("recent:", 0, end_index - 1)

        session_keys = []
        for token in tokens:
            session_keys.append("viewed:" + token)

        conn.delete(*session_keys)
        conn.hdel("login:", *tokens)
        conn.zrem("recent:", *tokens)


def clean_full_sessions(conn):
    while not QUIT:
        size = conn.zcard("recent:")
        if size <= LIMIT:
            time.sleep(1)
            continue

        end_index = min(size - LIMIT, 100)
        sessions = conn.zrange("recent:", 0, end_index - 1)

        session_keys = []
        for sess in sessions:
            session_keys.append("viewed:" + sess)
            session_keys.append("cart:" + sess)

        conn.delete(*session_keys)
        conn.hdel("login:", *sessions)
        conn.zrem("recent:", *sessions)



