"""This is for implementing Linked List data Structure"""
class Node:
    def __init__(self, value=None, next=None):
        self.value = value
        self.next = next

    def __str__(self):
        return 'Node ['+str(self.value)+']'

class LinkedList:
    def __init__(self):
        self.first = None
        self.last = None

    def insert(self, x):
        if self.first is None:
            self.first = Node(x, None)
            self.last = self.first
        elif self.last == self.first:
            self.last = Node(x, None)
            self.first.next = self.last
        else:
            c = Node(x, None)
            self.last.next = c
            self.last = c

    # Find some value in Link, if not exits, raise an error
    def _find_value(self, value):
        previous, current = self.first, self.first.next
        while current is not None:
            if current.value == value:
                return previous, current
            previous, current = current, current.next
        raise ValueError('Value %d is not in list'%(int(value)))

    # Drop some given value
    def drop_value(self, value):
        pre_node, match_node = self._find_value(value)
        pre_node.next = match_node.next


    # def remove_duplicate(self):
    #     uni_list = []
    #     if self.first.value is not None:
    #         curr_node = self.first
    #         uni_list.append(self.first.value)
    #         while curr_node.next is not None:
    #             if curr_node is self.first:
    #                 curr_node = curr_node.next
    #                 continue
    #             pre_node, curr_node = self._find_value(curr_node.value)
    #             print('P', pre_node.value)
    #             print('C', curr_node.value)
    #             if curr_node.value in uni_list:
    #                 pre_node.next = curr_node.next
    #             else:
    #                 curr_node = curr_node.next
    #                 uni_list.append(curr_node.value)

    def __str__(self):
        if self.first is not None:
            c = self.first
            out = 'Linked list ['+ str(c.value)+','
            while c.next is not None:
                c = c.next
                out += str(c.value)+','
            return out + ']'
        return 'Empty List'

    # This function is used to compute one linked list value with another list, so here is to convert
    # linked list to string
    def add_str(self):
        if self.first is not None:
            c = self.first
            out = str(c.value)
            while c.next is not None:
                c = c.next
                out += str(c.value)
            return out
        return ''

    def reset(self):
        self.__init__()


if __name__ == '__main__':
    l = LinkedList()
    l.insert(1)
    l.insert(1)
    l.insert(2)
    l.insert(3)
    l.insert(4)
    print(l)
    #print('drop duplicate:', l.remove_duplicate())
    # l.drop_value(2)
    # print('Drop one value result:',l)
    # print('drop duplicated result:',  l.remove_duplicate())

    f = LinkedList()
    f.insert(3)
    f.insert(1)

    # This is implement for using string to do the add function
    num1 = int(l.add_str())
    num2 = int(f.add_str())
    # print('num1:', num1)
    # print('num2:', num2)
    # print('added result:', (num1+num2))