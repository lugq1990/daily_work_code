# -*- coding:utf-8 -*-
"""This is to implement the linked list"""


class Node:
    def __init__(self, value):
        self.value = value if value is not None else None
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None
        self.nums = 0

    def push(self, value):
        self.nums += 1
        if self.head is None:
            self.head = Node(value)
        else:
            node = self.head
            while node.next is not None:
                node = node.next
            node.next = Node(value)
        return self

    def find(self, value):
        if self.head is None:
            raise ValueError("Nothing in the list!")

        node = self.head
        step = 0
        if node.value == value:
            return step
        else:
            while node.next is not None:
                step += 1
                node = node.next
                if node.value == value:
                    return step
            raise ValueError("Not found value %d in the list." % value)

    def insert(self, value, position=-1):
        self.nums += 1
        if position < -1:
            position = -1

        if self.head is None:
            self.head = Node(value)
        else:
            node = self.head
            curr_step = 0
            if position == 0:
                curr_head = self.head
                self.head = Node(value)
                self.head.next = curr_head
            elif position == -1:
                while node.next is not None:
                    node = node.next
                node.next = Node(value)
            else:
                while node.next is not None and curr_step < position:
                    node = node.next
                    curr_step += 1
                if node.next is None:
                    node.next = Node(value)
                else:
                    next_node = node.next
                    node.next = Node(value)
                    node.next.next = next_node
            return self

    def find(self, value):
        if self.head is None:
            raise ValueError("Nothing in the list!")
        else:
            curr_step = 0
            node = self.head
            if node.value == value:
                return curr_step
            else:
                while node.next is not None:
                    node = node.next
                    curr_step += 1
                    if node.value == value:
                        return curr_step
                raise ValueError("Couldn't find value %d in the list!" % value)

    def remove(self, value):
        if self.head is None:
            raise ValueError("Nothing in the list!")
        else:
            node = self.head
            while node.next is not None:
                if node.next.value == value:
                    node.next = node.next.next
                    return True
            raise ValueError("Value %d not in the list!" % value)

    def delete(self):
        self.head = None

    def reverse(self):
        pre, curr = None, self.head
        while curr is not None:
            curr.next, pre, curr = pre, curr, curr.next
        self.head = pre
        return self

    @property
    def length(self):
        return self.nums

    def print_var(self):
        if self.head is None:
            print('Nothing in the list!')
        else:
            node = self.head
            re = [node.value]
            while node.next is not None:
                node = node.next
                re.append(node.value)
            print('Get List: ', re)



if __name__ == '__main__':
    ll = LinkedList()
    ll.push(0)
    ll.insert(1, -1)
    ll.insert(2, 0)
    ll.insert(3, 1)
    ll.print_var()

    ll.insert(100, 5)
    ll.print_var()

    print('find: ', ll.find(2))

    ll.reverse()
    ll.print_var()

    ll.remove(1)
    ll.print_var()



