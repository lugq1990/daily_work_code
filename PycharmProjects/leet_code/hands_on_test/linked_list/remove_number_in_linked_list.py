# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        if not head:
            return head
        head.next = self.removeElements(head.next, val)
        if head.val == val:
            return head.next
        else:
            return head



if __name__ == '__main__':
    vals = [1, 2, 4, 2]
    val = 2

    head = ListNode()
    node = head
    for x in vals:
        node.next = ListNode(x)
        node = node.next

    head = head.next
    while head:
        print(head.val, sep='\t')
        head = head.next
    
    res = Solution().removeElements(head, val)
    while res:
        print(res.val, sep='\t')
        res = res.next