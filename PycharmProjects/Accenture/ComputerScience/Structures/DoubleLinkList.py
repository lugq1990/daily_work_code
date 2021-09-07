"""This is a implement for Double Linked List"""
class Node:
    def __init__(self, value=None, prev=None, next=None):
        self.value = value
        self.prev = prev
        self.next = next

    def __str__(self):
        return 'Node ['+str(self.value)+']'

class DoubleLinked:
    def __init__(self):
        self.first = None
        self.last = None

    def insert(self, x):
        if self.first is None:
            self.first = Node(x, None, None)
            self.last = self.first
        elif self.last == self.first:
            self.last = Node(x, None, None)
            self.first.next = self.last
            self.last.prev = self.first
        else:
            c = Node(x, None, None)
            self.last.next = c
            c.prev = self.last
            self.last = c

    def __str__(self):
        out = ''
        if self.first is not None:
            c = self.first
            out += 'Forward link:'+ str(c.value)+','
            while c.next is not None:
                c = c.next
                out += str(c.value)+','
            out += '|'
        if self.last is not None:
            c = self.last
            out += 'Backwark Link:'+str(c.value)+','
            while c.prev is not None:
                c = c.prev
                out += str(c.value) + ','
        return out

    def clear(self):
        self.__init__()

if __name__ == '__main__':
    dl = DoubleLinked()
    dl.insert(19)
    dl.insert(2)
    dl.insert(100000)
    dl.insert(3)
    print(dl)
    dl.clear()
    print(dl)

