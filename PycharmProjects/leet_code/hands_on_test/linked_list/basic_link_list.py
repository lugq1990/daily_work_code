"""Implement with both linked list and biderectional linked list"""

class Node:
    def __init__(self, val=None) -> None:
        self.val = val
        self.next = None
    

class LinkedList:
    def __init__(self) -> None:
        self.head = None

    def add(self, val):
        """Add a val into linked list"""
        if not self.head:
            self.head = Node(val)
        else:
            node = self.head
            while node.next:
                node = node.next
            node.next = Node(val)
        
        return self.head

    def search(self, val):
        if not self.head:
            print("Nothing in the linked list! couldn't find it")
            return -1
        else:
            node = self.head
            i = 0
            if node.val == val:
                print("Get val: {} get first node.")
                return i
            else:
                while node.next:
                    i += 1
                    if node.next.val == val:
                        print("Get val: {} get index: {}".format(val, i))
                        return i
                    else:
                        node = node.next
                return -1

    def delete(self, val):
        if not self.head:
            return -1
        else:
            node = self.head
            if node.val == val:
                next_node = node.next
                node = next_node
            else:
                while node.next:
                    if node.next.val == val:
                        next_node = node.next.next
                        node.next = next_node
                    else:
                        node = node.next

    def change(self, val, change_to):
        if not self.head:
            return 
        else:
            node = self.head
            if node.val == val:
                node.val = change_to
            else:
                while node.next:
                    if node.next.val == val:
                        node.next.val = change_to
                        return
                    else:
                        node = node.next
                return 

    def print(self):
        node = self.head
        if not node:
            print("Nothing in the node")
        else:
            while node:
                print(node.val)
                node = node.next
    

if __name__ == '__main__':
    ll = LinkedList()
    ll.add(1)
    ll.add(2)
    ll.add(3)
    ll.print()

    print(ll.search(2))

    print("*"*10)
    ll.delete(2)
    ll.print()
    
    print("*"*10)
    ll.change(3, 10)
    ll.print()

    