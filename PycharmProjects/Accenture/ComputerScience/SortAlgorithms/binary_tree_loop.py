from tkinter.messagebox import NO


class Node:
    def __init__(self, data) -> None:
        self.data = data
        self.left = None
        self.right = None
        

class BinaryTree:
    def __init__(self) -> None:
        self.head = None
    
    def insert(self, value):
        if not self.head:
            self.head = Node(value)
        else:
            node = self.head
            while node.left and node.right:
                v = node.data
                if value < v:
                    node = node.left
                else:
                    node = node.right
            
            if not node.left:
                node.left = Node(value)
            else:
                node.right = Node(value)
    
    def pre_print(self):
        node = self.head
        self._print_node(node)
    
    def _print_node(self, node):
        if not node:
            return
        print(node.data)
        self._print_node(node.left)
        self._print_node(node.right)
    
    def mid_print(self):
        node = self.head
        self._mid_print(node)
    
    def _mid_print(self, node):
        if not node:
            return
        
        self._mid_print(node.left)
        print(node.data)
        self._mid_print(node.right)
        
    def after_print(self):
        node = self.head
        self._after_print(node)
    
    def _after_print(self, node):
        if not node:
            return
        self._after_print(node.left)
        self._after_print(node.right)
        print(node.data)
        
    
if __name__ == "__main__":
    tree = BinaryTree()
    tree.insert(2)
    tree.insert(0)
    tree.insert(3)
    
    tree.pre_print()
    print()
    tree.mid_print()
    print()
    tree.after_print()