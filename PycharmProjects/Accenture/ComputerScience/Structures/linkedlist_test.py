# -*- coding:utf-8 -*-
"""this is to implement the linked list algorithm"""
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

class Linkedlist:
    def __init__(self):
        self.head = None

    def add(self, value, position=-1):
        if self.head is None:
            self.head = Node(value)
        else:
            # according to different position to add value
            if position == -1:
                node = self.head
                while node.next is not None:
                    node = node.next
                node.next = Node(value)
            elif position == 0:
                new_head = Node(value)
                curr_head = self.head
                self.head = new_head
                self.head.next = curr_head
            elif position >= 1:
                node = self.head
                step = 1
                if node.next is None:  # with just one node
                    node.next = Node(value)
                else:
                    while step <= position - 1 and node.next is not None:
                        step += 1
                        node = node.next
                    next_node = node.next
                    node.next = Node(value)
                    node.next.next = next_node
            else:
                raise ValueError("shouldn't provide with negative position!")

    def print_list(self):
        node = self.head
        if node is None:
            print('None list')
        else:
            res = [node.value]
            while node.next is not None:
                node = node.next
                res.append(node.value)
            print("Get list:", res)

    # here this function to reverse the linked list!
    def reverse(self):
        if self.head is None:
            raise ValueError("Couldn't reverse a None list")
        else:
            pre, curr = None, self.head
            while curr is not None:
                curr.next, pre, curr = pre, curr, curr.next
            self.head = pre
            return self


    @property
    def length(self):
        length = 0
        node = self.head
        if node is None:
            return 0
        else:
            length += 1
            while node.next is not None:
                node = node.next
                length += 1
            return length


if __name__ == '__main__':
    ll = Linkedlist()
    ll.add(1)
    ll.add(2)
    ll.add(3, -1)
    ll.add(4, 2)

    ll.print_list()

    print("Length:", ll.length)
    ll.reverse()
    ll.print_list()