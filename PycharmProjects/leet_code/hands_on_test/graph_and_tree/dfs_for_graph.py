"""Deep first to search a graph as DFS and Broad first to search as BFS."""

class Node:
    def __init__(self, val=None) -> None:
        self.val = val
        self.left = None
        self.right = None
        self.visited = False


class Tree:
    def __init__(self) -> None:
        self.root = None
    
    def add(self, val):
        if not self.root:
            self.root = Node(val)
        else:
            node = self.root
            while node:
                root_val = node.val
                if val <= root_val:
                    if not node.left:
                        node.left = Node(val)
                        return
                    else:
                        node = node.left
                else:
                    if not node.right:
                        node.right = Node(val)
                        return
                    else:
                        node = node.right
    
    def search(self, val):
        if not self.root:
            return False
        else:
            node = self.root
            while node:
                if node.val == val:
                    return True
                elif val >= node.val:
                    if node.right:
                        node = node.right
                    else:
                        return False
                else:
                    if node.left:
                        node = node.left
                    else:
                        return False

            return False

    def dfs_print(self, node=None):
        if not node:
            node = self.root
        
        print(node.val)
        node.visited = True
        node_list = []
        if node.left:
            node_list.append(node.left)
        if node.right:
            node_list.append(node.right)
        
        for node in node_list:
            if not node.visited:
                self.dfs_print(node)

    def bfs_print(self, node=None):
        if not node:
            node = self.root

        queue = []
        queue.append(node)
        print(node.val)
        node.visited = True

        while queue:
            out_node = queue.pop()
            if out_node.left:
                if not out_node.left.visited:
                    print(out_node.left.val)
                    out_node.left.visited = True
                    queue.append(out_node.left)

            if out_node.right:
                if not out_node.right.visited:
                    print(out_node.right.val)
                    out_node.right.visited = True
                    queue.append(out_node.right)

            
        

if __name__ == '__main__':
    tree = Tree()
    tree.add(19)
    tree.add(1)
    tree.add(20)
    tree.add(2)

    print(tree.search(10))

    # tree.dfs_print()
    print("*"*10)
    tree.bfs_print()

