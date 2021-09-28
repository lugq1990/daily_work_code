"""Implement with stack: LIFO without python list"""
class Node:
    def __init__(self, val=None) -> None:
        self.val = val
        self.next = None

class Stack:
    def __init__(self) -> None:
        self.head = None
    
    def add(self, val):
        new_node = Node(val)
        new_node.next = self.head
        self.head = new_node

    def pop(self):
        if self.head:
            val = self.head.val
            next_node = self.head.next
            self.head = next_node
            return val
        else:
            return None

    def top(self):
        if self.head:
            return self.head.val
        else:
            return None
    
    def is_null(self):
        if not self.head:
            return True
        else:
            return False
    
    

if __name__ == '__main__':
    stack = Stack()
    stack.add(1)
    stack.add(2)
    stack.add(3)

    print(stack.pop())

    print("*"*10)
    print(stack.top())
    print(stack.is_null())