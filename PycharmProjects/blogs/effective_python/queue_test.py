# *-* encoding: utf-8

# from queue import Queue
# from threading import Thread
# import time


# queue = Queue()


# def conumser():
#     time.sleep(.1)
#     print("consumer get 1")
#     queue.get()
#     print("consumer get 2")
#     queue.get()
#     queue.task_done()


# thread = Thread(target=conumser)
# thread.start()

# print("Producing")
# queue.put(object())
# print("Produce 1")
# queue.put(object())
# print("Produce 2")
# thread.join()
# print("producing done.")


def test_jython():
    import random

    print(random.random())


if __name__ == "__main__":
    test_jython()
    


from gensim import models