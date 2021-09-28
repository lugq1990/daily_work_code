"""implement queue without python list"""

class Node:
    def __init__(self, val=None) -> None:
        self.val = val
        self.next = None
    
    
class Queue:
    def __init__(self) -> None:
        self.head = None
        self.last = None

    def enqueue(self, val):
        if not self.head:
            self.head = Node(val)
            self.last = self.head
        else:
            node = self.last
            while node.next:
                node = node.next
            node.next = Node(val)

    def dequeue(self):
        if not self.head:
            return None
        else:
            val = self.head.val
            if self.head.next is None:
                self.head = None
                self.last = None
            else:
                self.head = self.head.next
            return val
    
    def top(self):
        if not self.head:
            return None
        else:
            val = self.head.val
            return val
    
    def last_val(self):
        if not self.last:
            return None
        else:
            return self.last.val

    def print(self):
        if not self.head:
            return None
        else:
            node = self.head
            while node:
                print(node.val, sep='\t')
                node = node.next


if __name__ == '__main__':
    q = Queue()

    q.enqueue(1) 
    q.enqueue(2)
    q.enqueue(3)

    print(q.dequeue())
    
    print("*"*10)
    q.print()

    print("*"*10)
    print(q.top())

    print("*"*10)
    print(q.last_val())
