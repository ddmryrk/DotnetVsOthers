class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def print(self):
        result = []
        current = self
        while current:
            result.append(str(current.val))
            current = current.next
        return ' -> '.join(result)
    
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
