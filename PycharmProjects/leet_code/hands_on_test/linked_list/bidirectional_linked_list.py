"""Bidirectional linked list implement"""

class Node:
    def __init__(self, val=None) -> None:
        self.val = val
        self.pre = None
        self.next = None


class BidiLinkedList:
    def __init__(self) -> None:
        self.head = None
        self.last = None

    def add(self, val):
        if not self.head:
            self.head = Node(val)
            self.last = self.head
        else:
            node = self.head
            while node.next:
                node = node.next
            new_node = Node(val)
            new_node.pre = node
            node.next = new_node

    def delete(self, val):
        if not self.head:
            return 
        else:
            node = self.head
            if node.val == val:
                next_node = node.next
                if next_node.next:
                    next_next_node = next_node.next
                    node = next_node
                    next_next_node.pre = node
                else:
                    self.head = None
                    self.last = None
            else:
                while node.next:
                    if node.next.val == val:
                        next_next_node = node.next.next
                        node.next = next_next_node
                        next_next_node.pre = node
                        return
                    else:
                        node = node.next
                return

    def search(self, val):
        if not self.head:
            return -1
        else:
            node = self.head
            i = 0
            if node.val == val:
                return i
            else:
                while node.next:
                    i += 1
                    if node.next.val == val:
                        return i
                    else:
                        node = node.next
                return -1
            
    def print(self):
        if not self.head:
            print("Nothing")
        else:
            node = self.head
            while node:
                print(node.val)
                node = node.next


if __name__ == '__main__':
    bll = BidiLinkedList()
    bll.add(1)
    bll.add(2)
    bll.add(3)

    bll.print()
    
    print("*" * 10)
    bll.delete(2)
    bll.print()

    print("*"*10)
    print(bll.search(3))
                
