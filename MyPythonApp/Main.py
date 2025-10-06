from solutions import *
from my_types import ListNode

class Solution:	
    def main(self):
        print(f"Indices: {two_sum(numbers=[2, 7, 11, 15], target=9)}")
  
        print(f"Is Valid: {is_valid_parentheses(s='()[]{}')}")

        list1 = ListNode(1, ListNode(2, ListNode(4)))
        list2 = ListNode(1, ListNode(3, ListNode(4)))
        print(f"Merged: {merge_two_sorted_lists(list1, list2).print()}")
  
        print(f"Sum: {add_binary(a='10', b='101')}")
  
        print(f"Sum: {climbStairs(n=11)}")

        print(f"Max Depth: {maxDepth(TreeNode(3, left=TreeNode(9), right=TreeNode(20, left=TreeNode(15), right=TreeNode(7))))}")
  
        print(maxProfit([7,2,6,1,4,3]))
        
        print(myAtoi("   -42"))

if __name__ == "__main__":
    Solution().main()