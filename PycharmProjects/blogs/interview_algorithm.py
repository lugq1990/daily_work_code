# 1. check Linked-list is cycled

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None
    
    def insert(self, value):
        if not self.head:
            self.head = Node(value)
        else:
            node = self.head
            while node.next:
                node = node.next
            node.next = Node(value)
        return self

    def print_linked_list(self):
        if not self.head:
            print("Nothing")
        else:
            node = self.head
            while node:
                print(node.value)
                node = node.next


def is_cycle(node):
    node1 = node.head
    node2 = node.head

    while node2 and node2.next:
        node1 = node1.next
        node2 = node2.next.next
        if node1 == node2:
            return True

    return False


def get_great_common_division(a, b):
    if a == b:
        return a
    big = a if a > b else b
    small = a if a < b else b
    # if (big % small == 0):
    #     return small
    return get_great_common_division(big - small, small)


class StackAsQueue:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def enqueue(self, value):
        self.stack1.append(value)
    
    def dequeue(self):
        if not self.stack2:
            if not self.stack1:
                return 
            while self.stack1:
                self.stack2.append(self.stack1.pop())

        return self.stack2.pop()

    def whole_values(self):
        v = self.dequeue()
        while v:
            print(v)
            v = self.dequeue()

def is_power(x):
    return (x & x -1) == 0


# 主方法
if __name__ == "__main__":
    ll = LinkedList()
    ll.insert(4)
    ll.insert(5)
    ll.insert(6)
    ll.insert(4)
    
    ll.print_linked_list()

    print(is_cycle(ll))

    print(get_great_common_division(100, 80))

    print(is_power(7))

    saq = StackAsQueue()
    saq.enqueue(1)
    saq.enqueue(2)
    saq.enqueue(4)
    saq.enqueue(5)

    saq.whole_values()
