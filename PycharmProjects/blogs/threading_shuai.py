import threading
import time
from queue import Queue  


class CumtomThread(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.__queue = queue
        
    def run(self):
        while True:
            f_queue = self.__queue.get()
            f_queue()
            self.__queue.task_done()

class CumtomSepThread(threading.Thread):
    def __init__(self, thread_id, name, counter):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.name = name
        self.counter = counter

    def run(self):
        print("start thread: " + self.name)
        get_fish(self.name, 0, self.counter)
        print("End of thread:" + self.name)



def f():
    print("Start to do: {}".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))


def get_fish(name, delay, counter):
    while counter:
        time.sleep(delay)
        print("Start to get fish: {} at time: {}".format(name, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        counter -=1



def queue_pool():
    queue_pool = Queue(5)
    for i in range(queue_pool.maxsize):
        t = CumtomThread(queue_pool)
        t.setDaemon(True)
        t.start()
    
    name_list = ['lu', 'liu']
    fish_num = [10, 20]
    for name, number in zip(*[name_list, fish_num]):
        queue_pool.put(get_fish(name, 0, number))

    queue_pool.join()


if __name__ == '__main__':
    lu_thread = CumtomSepThread(1, 'lu', 10)
    liu_thread = CumtomSepThread(2, 'liu', 10)

    lu_thread.start()
    liu_thread.start()

    lu_thread.join()
    liu_thread.join()

    print("End of main thread")

    from gensim.models import Word2Vec