# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def removeDuplicateNodes(self, head: ListNode) -> ListNode:
        n_set = set()

        node = head
        while node:
            val = node.val
            if val not in n_set:
                n_set.add(val)
                node = node.next
            else:
                if node.next:
                    node.next = node.next.next
                else:
                    node.next = None
                node = node.next
        return head


if __name__ == '__main__':
    vals = [1, 2, 3, 3, 2, 1]
    val = 2

    head = ListNode()
    node = head
    for x in vals:
        node.next = ListNode(x)
        node = node.next

    head = head.next
    
    res = Solution().removeDuplicateNodes(head)
    while res:
        print(res.val, sep='\t')
        res = res.next