from my_types import ListNode, TreeNode
from typing import List, Optional

def two_sum(numbers, target):
    num_dict = {}
    for i, num in enumerate(numbers):
        complement = target - num
        if complement in num_dict:
            return (num_dict[complement], i)
        num_dict[num] = i
    return None

def is_valid_parentheses(s: str) -> bool:
    i=0
    a=[]
    for i in range(len(s)):
        if s[i]=='('or s[i]=='['or s[i]=='{':
            a.append(s[i])
        else:
            if not a:
                return False
            top=a.pop()
            if s[i]==')'and top!='(':
                return False
            if s[i]==']'and top!='[':
                return False
            if s[i]=='}'and top!='{':
                return False
    return len(a)==0

def merge_two_sorted_lists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode()
    cur = dummy

    while list1 and list2:
        if list1.val > list2.val:
            cur.next = list2
            list2 = list2.next
        else:
            cur.next = list1
            list1 = list1.next
        
        cur = cur.next
    
    if list1:
        cur.next = list1
    else:
        cur.next = list2
    
    return dummy.next

def add_binary(a: str, b: str) -> str:
    s = []
    carry = 0
    i = len(a) - 1
    j = len(b) - 1

    while i >= 0 or j >= 0 or carry:
      if i >= 0:
        carry += int(a[i])
        i -= 1
      if j >= 0:
        carry += int(b[j])
        j -= 1
      s.append(str(carry % 2))
      carry //= 2

    return ''.join(reversed(s))

def climbStairs(n: int) -> int:
        #O(n) time and O(1) space complexity
        #if n == 0 or n == 1:
        #    return 1
        #return self.climbStairs(n-1) + self.climbStairs(n-2)
        if n == 0 or n == 1:
            return 1
        prev, curr = 1, 1
        for i in range(2, n+1):
            temp = curr
            curr = prev + curr
            prev = temp
        return curr
    
def maxDepth(root: Optional[TreeNode]) -> int:
    if not root:
        return 0
    
    return 1 + max(maxDepth(root.left), maxDepth(root.right))

def maxProfit(prices: List[int]) -> int:
        buy_price = prices[0]
        profit = 0

        for p in prices[1:]:
            if buy_price > p:
                buy_price = p
            
            profit = max(profit, p - buy_price)
        
        return profit

def myAtoi(s: str) -> int:
        s = s.strip()
        if not s:
            return 0

        i, num, sign = 0, 0, 1

        if s[0] == '+' or s[0] == '-':
            sign = -1 if s[0] == '-' else 1
            i += 1

        while i < len(s) and s[i].isdigit():
            digit = int(s[i])
            if num > (2**31 - 1 - digit) // 10:
                return 2**31 - 1 if sign == 1 else -2**31
            num = num * 10 + digit
            i += 1

        return sign * num