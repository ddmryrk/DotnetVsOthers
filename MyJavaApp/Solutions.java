
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;
import java.util.TreeSet;

public class Solutions {

    public int[] twoSum(int[] nums, int target) {
        // O(n) time complexity
        // O(n) space complexity

        Map<Integer, Integer> pairs = new HashMap<>();

        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];

            if (pairs.containsKey(complement)) {
                return new int[]{pairs.get(complement), i};
            }

            pairs.put(nums[i], i);
        }

        return new int[]{};
    }

    public boolean validParentheses(String s) {
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

    public ListNode mergeTwoSortedLists(ListNode list1, ListNode list2) {
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

    public String addBinary(String a, String b) {
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

    public int climbStairs(int n) {
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

    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        backtrack(nums, 0, res);
        return res;
    }

    private void backtrack(int[] nums, int start, List<List<Integer>> res) {
        if (start == nums.length) {
            res.add(arrayToList(nums));
            return;
        }

        for (int i = start; i < nums.length; i++) {
            swap(nums, start, i);
            backtrack(nums, start + 1, res);
            swap(nums, start, i);
        }
    }

    private List<Integer> arrayToList(int[] arr) {
        List<Integer> list = new ArrayList<>();
        for (int num : arr) {
            list.add(num);
        }
        return list;
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public int maxSubArray(int[] nums) {
        // on o1
        int max = nums[0];
        int current = nums[0];
        for (int i = 1; i < nums.length; i++) {
            current = Math.max(nums[i], current + nums[i]);
            max = Math.max(max, current);
        }
        return max;
    }

    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> result = new ArrayList<>();

        int top = 0;
        int left = 0;
        int bottom = matrix.length - 1;
        int right = matrix[0].length - 1;

        while (top <= bottom && left <= right) {
            // left->right
            for (int i = left; i <= right; i++) {
                result.add(matrix[top][i]);
            }
            top++;

            // top->bottom
            for (int i = top; i <= bottom; i++) {
                result.add(matrix[i][right]);
            }
            right--;

            // right->left
            if (top <= bottom) {
                for (int i = right; i >= left; i--) {
                    result.add(matrix[bottom][i]);
                }
                bottom--;
            }

            // bottom->top
            if (left <= right) {
                for (int i = bottom; i >= top; i--) {
                    result.add(matrix[i][left]);
                }
                left++;
            }
        }

        return result;
    }

    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));

        List<int[]> merged = new ArrayList<>();

        int[] prev = intervals[0];

        for (int i = 1; i < intervals.length; i++) {
            int[] interval = intervals[i];

            if (interval[0] <= prev[1]) {
                prev[1] = Math.max(prev[1], interval[1]);
            } else {
                merged.add(prev);
                prev = interval;
            }
        }

        merged.add(prev);

        return merged.toArray(new int[merged.size()][]);
    }

    public int[][] insert(int[][] intervals, int[] newInterval) {

        // on on
        List<int[]> result = new ArrayList<>();
        int i = 0;

        // Add all intervals that come **before** the new interval
        // Iterate through intervals and add non-overlapping intervals before
        // newInterval
        while (i < intervals.length && intervals[i][1] < newInterval[0]) {
            result.add(intervals[i]);
            i++;
        }

        // **Merge** the new interval with all overlapping intervals
        // Merge overlapping intervals
        while (i < intervals.length && intervals[i][0] <= newInterval[1]) {
            newInterval[0] = Math.min(newInterval[0], intervals[i][0]);
            newInterval[1] = Math.max(newInterval[1], intervals[i][1]);
            i++;
        }
        // Add merged newInterval
        result.add(newInterval);

        // Add all intervals that come **after** the merged interval
        // Add non-overlapping intervals after newInterval
        while (i < intervals.length) {
            result.add(intervals[i]);
            i++;
        }

        return result.toArray(new int[result.size()][]);
    }

    public int uniquePaths(int m, int n) {
        // Create a memoization table to store computed results
        int[][] memo = new int[m][n];

        // Initialize the memoization table with -1 to indicate uncomputed results
        for (int i = 0; i < m; i++) {
            Arrays.fill(memo[i], -1);
        }

        // Call the recursive function to compute unique paths
        var result = uniquePathsRecursive(0, 0, m, n, memo);

        return result;
    }

    public int uniquePathsRecursive(int x, int y, int m, int n, int[][] memo) {
        // If we reach the destination (bottom-right corner), return 1
        if (x == m - 1 && y == n - 1) {
            return 1;
        }

        // If we have already computed the result for this cell, return it from the memo
        // table
        if (memo[x][y] != -1) {
            return memo[x][y];
        }

        // Calculate the number of unique paths by moving right and down
        int rightPaths = 0;
        int downPaths = 0;

        // Check if it's valid to move right
        if (x < m - 1) {
            rightPaths = uniquePathsRecursive(x + 1, y, m, n, memo);
        }

        // Check if it's valid to move down
        if (y < n - 1) {
            downPaths = uniquePathsRecursive(x, y + 1, m, n, memo);
        }

        // Store the result in the memo table and return it
        memo[x][y] = rightPaths + downPaths;
        return memo[x][y];
    }

    public int uniquePaths2(int m, int n) {
        // Create a 2D DP array filled with zeros
        int[][] dp = new int[m][n];

        // Initialize the rightmost column and bottom row to 1
        for (int i = 0; i < m; i++) {
            dp[i][n - 1] = 1;
        }
        for (int j = 0; j < n; j++) {
            dp[m - 1][j] = 1;
        }

        // Fill in the DP array bottom-up
        for (int i = m - 2; i >= 0; i--) {
            for (int j = n - 2; j >= 0; j--) {
                dp[i][j] = dp[i + 1][j] + dp[i][j + 1];
            }
        }

        // Return the result stored in the top-left corner
        return dp[0][0];
    }

    public void sortColors(int[] nums) {

        int l = 0, m = 0, r = nums.length - 1;

        while (m <= r) {
            if (nums[m] == 0) {
                swap(nums, m++, l++);
            } else if (nums[m] == 1) {
                m++;
            } else {
                swap(nums, m, r--);
            }
        }
    }

    public static List<List<Integer>> subsets2(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        result.add(new ArrayList<>()); // başlangıçta boş küme

        for (int num : nums) {
            int size = result.size();
            // mevcut tüm subsetlere bu elemanı ekleyerek yeni subsetler oluştur
            for (int i = 0; i < size; i++) {
                List<Integer> newSubset = new ArrayList<>(result.get(i));
                newSubset.add(num);
                result.add(newSubset);
            }
        }

        return result;
    }

    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> list = new ArrayList<>();
        Arrays.sort(nums);
        backtrack(list, new ArrayList<>(), nums, 0);
        return list;
    }

    private void backtrack(List<List<Integer>> list, List<Integer> tempList, int[] nums, int start) {
        list.add(new ArrayList<>(tempList));
        for (int i = start; i < nums.length; i++) {
            tempList.add(nums[i]);
            backtrack(list, tempList, nums, i + 1);
            tempList.remove(tempList.size() - 1);
        }
    }

    public boolean exist(char[][] board, String word) {
        int m = board.length;
        int n = board[0].length;

        boolean[][] visited = new boolean[m][n];
        boolean result = false;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == word.charAt(0)) {
                    result = backtrack(board, word, visited, i, j, 0);
                    if (result) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    private boolean backtrack(char[][] board, String word, boolean[][] visited, int i, int j, int index) {
        if (index == word.length()) {
            return true;
        }

        if (i < 0
                || i >= board.length
                || j < 0
                || j >= board[0].length
                || visited[i][j]
                || board[i][j] != word.charAt(index)) {
            return false;
        }

        visited[i][j] = true;

        if (backtrack(board, word, visited, i + 1, j, index + 1)
                || backtrack(board, word, visited, i - 1, j, index + 1)
                || backtrack(board, word, visited, i, j + 1, index + 1)
                || backtrack(board, word, visited, i, j - 1, index + 1)) {
            return true;
        }

        visited[i][j] = false;
        return false;
    }

    public boolean isValidBST(TreeNode root) {
        return valid(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    private boolean valid(TreeNode node, long minimum, long maximum) {
        if (node == null) {
            return true;
        }

        if (!(node.val > minimum && node.val < maximum)) {
            return false;
        }

        return valid(node.left, minimum, node.val) && valid(node.right, node.val, maximum);
    }

    private int count = 0;
    private int result = 0;

    public int kthSmallest(TreeNode root, int k) {
        inorder(root, k);
        return result;
    }

    private void inorder(TreeNode node, int k) {
        if (node == null || count >= k) {
            return;
        }
        inorder(node.left, k);
        count++;
        if (count == k) {
            result = node.val;
            return;
        }
        inorder(node.right, k);
    }

    public int[][] insertBad(int[][] intervals, int[] newInterval) {
        List<int[]> list = new ArrayList<>(Arrays.asList(intervals));
        list.add(newInterval);

        // Sort + Merge → gereksiz
        list.sort((a, b) -> Integer.compare(a[0], b[0]));

        List<int[]> merged = new ArrayList<>();
        int[] current = list.get(0);
        for (int i = 1; i < list.size(); i++) {
            int[] next = list.get(i);
            if (next[0] <= current[1]) {
                current[1] = Math.max(current[1], next[1]);
            } else {
                merged.add(current);
                current = next;
            }
        }
        merged.add(current);
        return merged.toArray(new int[merged.size()][]);
    }

    public int[][] insertGood(int[][] intervals, int[] newInterval) {
        List<int[]> result = new ArrayList<>();
        int i = 0;
        int n = intervals.length;

        // Tüm çakışmayan interval'ları ekle
        while (i < n && intervals[i][1] < newInterval[0]) {
            result.add(intervals[i++]);
        }

        // Çakışan interval'ları merge et
        while (i < n && intervals[i][0] <= newInterval[1]) {
            newInterval[0] = Math.min(newInterval[0], intervals[i][0]);
            newInterval[1] = Math.max(newInterval[1], intervals[i][1]);
            i++;
        }
        result.add(newInterval);

        // Kalan interval'ları ekle
        while (i < n) {
            result.add(intervals[i++]);
        }

        return result.toArray(new int[result.size()][]);
    }

    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        Map<String, String> parent = new HashMap<>();
        Map<String, String> owner = new HashMap<>();

        for (List<String> acc : accounts) {
            String name = acc.get(0);
            for (int i = 1; i < acc.size(); i++) {
                parent.put(acc.get(i), acc.get(i));
                owner.put(acc.get(i), name);
            }
        }

        for (List<String> acc : accounts) {
            String root = find(parent, acc.get(1));
            for (int i = 2; i < acc.size(); i++) {
                parent.put(find(parent, acc.get(i)), root);
            }
        }

        Map<String, TreeSet<String>> unions = new HashMap<>();
        for (String email : parent.keySet()) {
            String root = find(parent, email);
            unions.computeIfAbsent(root, k -> new TreeSet<>()).add(email);
        }

        List<List<String>> result = new ArrayList<>();
        for (String root : unions.keySet()) {
            List<String> acc = new ArrayList<>();
            acc.add(owner.get(root));
            acc.addAll(unions.get(root));
            result.add(acc);
        }
        return result;
    }

    private String find(Map<String, String> parent, String s) {
        if (!parent.get(s).equals(s)) {
            parent.put(s, find(parent, parent.get(s)));
        }
        return parent.get(s);
    }

    public List<List<String>> accountsMerge2(List<List<String>> accounts) {
        Map<String, Set<String>> adj = new HashMap<>();

        for (List<String> account : accounts) {
            String email1 = account.get(1);
            for (int i = 2; i < account.size(); i++) {
                String curEmail = account.get(i);
                adj.computeIfAbsent(email1, k -> new HashSet<>()).add(curEmail);
                adj.computeIfAbsent(curEmail, k -> new HashSet<>()).add(email1);
            }
        }
        Set<String> visited = new HashSet<>();
        List<List<String>> result = new ArrayList<>();

        for (List<String> account : accounts) {
            String email1 = account.get(1);
            if (visited.contains(email1)) {
                continue;
            }
            ArrayList<String> mergedEmails = new ArrayList<>();
            dfs(email1, adj, mergedEmails, visited);
            if (mergedEmails.size() > 1) {
                Collections.sort(mergedEmails);
            }
            mergedEmails.add(0, account.get(0));//add name
            result.add(mergedEmails);
        }
        return result;
    }

    void dfs(String email, Map<String, Set<String>> adj, List<String> mergedEmails, Set<String> visited) {
        if (visited.contains(email)) {
            return;
        }
        mergedEmails.add(email);
        visited.add(email);
        for (String nextEmail : adj.getOrDefault(email, Collections.emptySet())) {
            dfs(nextEmail, adj, mergedEmails, visited);
        }
    }

    public int coinChange(int[] coins, int amount) {
        int max = amount + 1; // Represents infinity
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, max);
        dp[0] = 0; // 0 coins needed for amount 0

        for (int coin : coins) {
            for (int i = coin; i <= amount; i++) {
                dp[i] = Math.min(dp[i], dp[i - coin] + 1);
            }
        }

        return dp[amount] == max ? -1 : dp[amount];
    }

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();  // 📚 Create photo album for levels
        traversal(root, 0, res);                      // 🎬 Start filming from level 0
        return res;                                   // 🎁 Return the complete movie
    }

    public void traversal(TreeNode root, int lvl, List<List<Integer>> res) {
        if (root == null) {
            return;                     // 🚫 Empty scene - stop filming
        }
        // 🎯 STEP 1: Prepare frame for current level
        if (res.size() <= lvl) {
            res.add(new ArrayList());                 // 📸 Add new photo frame if needed
        }

        // 🎯 STEP 2: Capture current node in its level
        res.get(lvl).add(root.val);                   // 👑 Place king/queen in their row

        // 🎯 STEP 3: Film next generation (recursively)
        traversal(root.left, lvl + 1, res);           // 👶 Film left lineage
        traversal(root.right, lvl + 1, res);          // 👶 Film right lineage
    }

    
}
