import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

public class Solutions {
    public static int[] twoSum(int[] nums, int target) {
        // O(n) time complexity
        // O(n) space complexity

        Map<Integer, Integer> pairs = new HashMap<>();

        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];

            if (pairs.containsKey(complement)) {
                return new int[] { pairs.get(complement), i };
            }

            pairs.put(nums[i], i);
        }

        return new int[] {};
    }

    public static boolean validParentheses(String s) {
        // O(n) time complexity
        // O(n) space complexity
        // Stack is LIFO data structure

        Stack<Character> stack = new Stack<>();
        for (char ch : s.toCharArray()) {
            if (ch == '(' || ch == '[' || ch == '{') {
                stack.push(ch);
            } else {
                if (stack.isEmpty()) {
                    return false;
                }
                char top = stack.pop();
                if (ch == ')' && top != '(') {
                    return false;
                }
                if (ch == ']' && top != '[') {
                    return false;
                }
                if (ch == '}' && top != '{') {
                    return false;
                }
            }
        }
        return stack.isEmpty();
    }

    public static ListNode mergeTwoSortedLists(ListNode list1, ListNode list2) {
        // O(n + m) time complexity
        // O(1) space complexity
        ListNode dummy = new ListNode();
        ListNode cur = dummy;

        while (list1 != null && list2 != null) {
            if (list1.val > list2.val) {
                cur.next = list2;
                list2 = list2.next;
            } else {
                cur.next = list1;
                list1 = list1.next;
            }
            cur = cur.next;
        }

        cur.next = (list1 != null) ? list1 : list2;

        return dummy.next;
    }

    public static String addBinary(String a, String b) {
        // O(n) time and space complexity
        StringBuilder result = new StringBuilder();
        int i = a.length() - 1, j = b.length() - 1, carry = 0;

        while (i >= 0 || j >= 0 || carry > 0) {
            int sum = carry;
            if (i >= 0) {
                sum += a.charAt(i--) - '0';
            }
            if (j >= 0) {
                sum += b.charAt(j--) - '0';
            }
            result.append(sum % 2);
            carry = sum / 2;
        }

        return result.reverse().toString();
    }

    public static int climbStairs(int n) {
        // O(n) time and O(1) space complexity
        /*
         * if (n == 0 || n == 1) {
         * return 1;
         * }
         * return ClimbStairs(n-1) + ClimbStairs(n-2);
         */

        if (n == 0 || n == 1) {
            return 1;
        }
        int prev = 1, curr = 1;
        for (int i = 2; i <= n; i++) {
            int temp = curr;
            curr = prev + curr;
            prev = temp;
        }
        return curr;
    }

    public static int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }
    
}
