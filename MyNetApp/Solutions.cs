using System.Text;

namespace MyNetApp;

public class Solutions
{



    public static bool IsValid(string parentheses)
    {
        Stack<char> stack = new();

        Dictionary<char, char> dict = new(){
         {'(', ')'},
         {'{', '}'},
         {'[', ']'}
     };

        for (int i = 0; i < parentheses.Length; i++)
        {
            var current = parentheses[i];
            if (dict.Keys.Contains(current))
                stack.Push(current);
            else if (stack.Count > 0 && current == dict[stack.Peek()])
                stack.Pop();
            else
                return false;
        }

        return stack.Count == 0;
    }

    public static ListNode MergeTwoLists(ListNode list1, ListNode list2)
    {
        var dummy = new ListNode(0);
        var prev = dummy;

        ListNode p1 = list1, p2 = list2;
        while (p1 != null && p2 != null)
        {
            if (p1.val < p2.val)
            {
                prev.next = p1;
                p1 = p1.next;
            }
            else
            {
                prev.next = p2;
                p2 = p2.next;
            }
            prev = prev.next;
        }

        prev.next = p1 ?? p2;

        return dummy.next;
    }

    public static string AddBinary(string a, string b)
    {
        int aIndex = a.Length - 1;
        int bIndex = b.Length - 1;
        int carry = 0;
        var result = new System.Text.StringBuilder();

        while (aIndex >= 0 || bIndex >= 0 || carry > 0)
        {
            int bitA = (aIndex >= 0) ? a[aIndex] - '0' : 0;
            int bitB = (bIndex >= 0) ? b[bIndex] - '0' : 0;

            int sum = bitA + bitB + carry;
            result.Insert(0, (sum % 2).ToString());
            carry = sum / 2;

            aIndex--;
            bIndex--;
        }

        return result.ToString();
    }

    public static int ClimbStairs(int n)
    {
        // Tabulation
        /*
        var tab = new int[n + 1];
        if (tab.Length > 0) tab[0] = 1;
        if (tab.Length > 1) tab[1] = 1;
        for (var i = 2; i < tab.Length; i++)
            tab[i] = tab[i - 1] + tab[i - 2];
        return tab[n];
        */

        //Space Optimization
        if (n <= 1) return 1;
        int firstNum = 1, secondNum = 1, thirdNum = 0;
        for (int i = 2; i < n + 1; i++)
        {
            thirdNum = firstNum + secondNum;
            firstNum = secondNum;
            secondNum = thirdNum;
        }
        return thirdNum;
    }

    public static int MaxDepth(TreeNode root)
    {
        if (root == null)
            return 0;
        //Depth-First Search
        /*
        var leftMaxDepth = MaxDepth(root.left);
        var rightMaxDepth = MaxDepth(root.right);

        return Math.Max(leftMaxDepth, rightMaxDepth) + 1;
        */

        //Breadth-First Search
        var queue = new Queue<TreeNode>();
        queue.Enqueue(root);

        var depth = 0;

        while (queue.Count != 0)
        {
            var size = queue.Count;

            for (var i = 0; i < size; i++)
            {
                var current = queue.Dequeue();

                if (current.left != null)
                {
                    queue.Enqueue(current.left);
                }

                if (current.right != null)
                {
                    queue.Enqueue(current.right);
                }
            }

            depth++;
        }

        return depth;

    }

    public static bool IsBalanced(TreeNode root)
    {
        if (root == null)
        {
            return true;
        }

        int differHigh = int.Abs(TreeHeight(root.left) - TreeHeight(root.right));
        if (differHigh > 1)
        {
            return false;
        }

        return IsBalanced(root.left) && IsBalanced(root.right);
    }
    private static int TreeHeight(TreeNode root)
    {
        if (root == null)
        {
            return 0;
        }

        return int.Max(TreeHeight(root.left), TreeHeight(root.right)) + 1;
    }

    public static int MaxProfit(int[] prices)
    {
        //Two Pointers O(n) & O(1)
        //var maxProfit = 0;
        //var left = 0;
        //var right = 1;

        //while (right < prices.Length)
        //{
        //    if (prices[left] < prices[right])
        //    {
        //        maxProfit = Math.Max(maxProfit, prices[right] - prices[left]);
        //    }
        //    else
        //    {
        //        left = right;
        //    }

        //    right++;
        //}

        //return maxProfit;

        //Dynamic Programming  O(n) & O(1)
        int minPrice = int.MaxValue;
        int maxProfit = 0;

        foreach (int currentPrice in prices)
        {
            minPrice = Math.Min(currentPrice, minPrice);
            maxProfit = Math.Max(maxProfit, currentPrice - minPrice);
        }

        return maxProfit;
    }

    public static bool IsPalindrome(string s)
    {
        for (int i = 0, j = s.Length - 1; j > i;)
        {
            if (!char.IsLetterOrDigit(s[i]))
            {
                i++;
                continue;
            }

            if (!char.IsLetterOrDigit(s[j]))
            {
                j--;
                continue;
            }

            if (char.ToLower(s[i++]) != char.ToLower(s[j--]))
            {
                return false;
            }
        }
        return true;
    }

    public static bool HasCycle(ListNode head)
    {
        //Two-Pointers on o1
        ListNode slow = head;
        ListNode fast = head;

        while (fast != null && fast.next != null)
        {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast)
            {
                return true;
            }
        }
        return false;

        //Hash Table on on
        /*
        HashSet<ListNode> visited_nodes = new HashSet<ListNode>();
        ListNode current_node = head;
        while (current_node != null) {
            if (visited_nodes.Contains(current_node)) {
                return true;
            }
            visited_nodes.Add(current_node);
            current_node = current_node.next;
        }
        return false;
        */
    }

    public static int MajorityElement(int[] nums)
    {
        int count = 0;
        int candidate = 0;
        foreach (var num in nums)
        {
            if (count == 0)
            {
                candidate = num;
            }
            count += (num == candidate) ? 1 : -1;
        }
        return candidate;
    }

    public static ListNode ReverseList(ListNode head)
    {
        //ListNode prev = null;
        //ListNode current = head;
        //ListNode next = null;

        //while (current != null)
        //{
        //    next = current.next;
        //    current.next = prev;
        //    prev = current;
        //    current = next;
        //}
        //return prev;

        //on o1
        ListNode resultNode = null;
        while (head != null)
        {
            resultNode = new ListNode(head.val, resultNode);
            head = head.next;
        }
        return resultNode;
    }

    public static bool ContainsDuplicate(int[] nums)
    {
        //o(n) o(n)
        //return new HashSet<int>(nums).Count < nums.Length;  

        //o(n log n) o(1)
        Array.Sort(nums);

        int latest = nums[0];
        for (var i = 1; i < nums.Length; i++)
        {
            if (latest == nums[i])
                return true;
            latest = nums[i];
        }
        return false;
    }

    public static TreeNode InvertTree(TreeNode root)
    {
        if (root?.left == null && root?.right == null)
            return root;

        var temp = new TreeNode(root.val,
            left: InvertTree(root.right),
            right: InvertTree(root.left));

        return temp;
    }

    //Implement Queue using Stacks
    public class MyQueue
    {
        private Stack<int> inputStack;
        private Stack<int> outputStack;
        public MyQueue()
        {
            inputStack = new Stack<int>();
            outputStack = new Stack<int>();
        }

        public void Push(int x)
        {
            // Move elements from outputStack to inputStack
            while (outputStack.Count > 0)
            {
                inputStack.Push(outputStack.Pop());
            }

            // Push the new element onto inputStack
            inputStack.Push(x);

            // Move elements from inputStack to outputStack
            while (inputStack.Count > 0)
            {
                outputStack.Push(inputStack.Pop());
            }
        }

        public int Pop()
        {
            return outputStack.Pop();
        }

        public int Peek()
        {
            return outputStack.Peek();
        }

        public bool Empty()
        {
            return outputStack.Count == 0;
        }
    }

    public static bool IsAnagram(string s, string t)
    {
        //var sArray = s.ToCharArray();
        //var tArray = t.ToCharArray();

        //Array.Sort(sArray);
        //Array.Sort(tArray);

        //return new string(sArray) == new string(tArray);

        if (s.Length != t.Length) return false;

        var count = new Dictionary<char, int>();
        foreach (char c in s)
            if (count.ContainsKey(c)) count[c]++;
            else count[c] = 1;

        foreach (char c in t)
        {
            if (!count.ContainsKey(c)) return false;
            count[c]--;
            if (count[c] == 0) count.Remove(c);
        }

        return count.Count == 0;
        /*
        int[] count = new int[26];

        for (int i = 0; i < s.Length; i++) {
            count[s[i] - 'a']++;
            count[t[i] - 'a']--;
        }

        foreach (int c in count) {
            if (c != 0) {
                return false;
            }
        }

        return true;
        */
    }

    public static int FirstBadVersion(int n)
    {
        int left = 1;
        int right = n;
        while (left <= right)
        {
            int middle = (right - left) / 2 + left;
            if (!IsBadVersion(middle))
            {
                left = middle + 1;
            }
            else
            {
                right = middle - 1;
            }
        }
        return left;
    }
    private static bool IsBadVersion(int version)
    {
        // The isBadVersion API is defined in the parent class VersionControl.
        // 9 is to simulate API's response
        return version >= 9;
    }

    public static bool CanConstruct(string ransomNote, string magazine)
    {
        //a = 97
        //b = 98
        //x = 120
        //y = 121
        //z = 122
        int[] letters = new int[26];
        foreach (char c in magazine)
            letters[c - 'a']++;
        foreach (char ch in ransomNote)
        {
            letters[ch - 'a']--;
            if (letters[ch - 'a'] == -1)
                return false;
        }
        return true;
    }

    public static int LongestPalindrome(string s)
    {
        //var a = s.ToLower().ToCharArray();
        //Array.Sort(a);

        //int longest = 0;
        //bool hasSingle = false;

        //for (int i = 0; i < a.Length; i++)
        //{
        //    if (i == a.Length - 1)
        //    {
        //        hasSingle = true;
        //        break;
        //    }

        //    if (a[i] == a[i + 1])
        //    {
        //        longest += 2;
        //        i++;
        //    }
        //    else
        //    {
        //        hasSingle = true;
        //    }
        //}

        //return hasSingle ? longest + 1 : longest;

        var set = new HashSet<char>();
        var maxLength = 0;

        foreach (var c in s)
        {
            if (set.Contains(c))
            {
                set.Remove(c);
                maxLength += 2;
            }
            else
                set.Add(c);
        }

        return set.Count > 0 ? maxLength + 1 : maxLength;
    }

    public static int DiameterOfBinaryTree(TreeNode root)
    {
        int diameter = 0;
        Dfs(root, ref diameter);
        return diameter;
    }
    private static int Dfs(TreeNode node, ref int diameter)
    {
        if (node == null) return 0;

        int leftHeight = Dfs(node.left, ref diameter);
        int rightHeight = Dfs(node.right, ref diameter);

        diameter = Math.Max(diameter, leftHeight + rightHeight);

        return 1 + Math.Max(leftHeight, rightHeight);
    }

    public static int Search(int[] nums, int target)
    {
        return BinarySearch(nums, target, 0, nums.Length - 1);
    }
    private static int BinarySearch(int[] nums, int target, int left, int right)
    {
        if (left > right)
            return -1;
        var mid = left + (right - left) / 2;
        if (nums[mid] == target)
            return mid;
        if (nums[mid] > target)
            return BinarySearch(nums, target, left, mid - 1);
        else
            return BinarySearch(nums, target, mid + 1, right);
    }

    public static int[][] FloodFill(int[][] image, int sr, int sc, int newColor)
    {
        int color = image[sr][sc];
        if (color != newColor)
        {
            dfs(image, sr, sc, color, newColor);
        }
        return image;
    }
    private static void dfs(int[][] image, int r, int c, int color, int newColor)
    {
        if (image[r][c] == color)
        {
            image[r][c] = newColor;
            if (r >= 1)
            {
                dfs(image, r - 1, c, color, newColor);//UP
            }
            if (c >= 1)
            {
                dfs(image, r, c - 1, color, newColor);//LEFT
            }
            if (r + 1 < image.Length)
            {
                dfs(image, r + 1, c, color, newColor);//DOWN
            }
            if (c + 1 < image[0].Length)
            {
                dfs(image, r, c + 1, color, newColor);//RIGHT
            }
        }
    }

    public static int[][] FloodFill2(int[][] image, int sr, int sc, int color)
    {
        int rows = image.Length;
        int cols = image[0].Length;
        FloodFill2(image, sr, sc, image[sr][sc], color, rows, cols);
        return image;
    }
    private static void FloodFill2(int[][] image, int row, int col, int oldColor, int newColor, int rows, int cols)
    {
        if (row < 0 || row >= rows || col < 0 || col >= cols ||
            image[row][col] != oldColor ||
            image[row][col] == newColor)
        {
            return;
        }

        image[row][col] = newColor;
        FloodFill2(image, row + 1, col, oldColor, newColor, rows, cols);
        FloodFill2(image, row, col + 1, oldColor, newColor, rows, cols);
        FloodFill2(image, row - 1, col, oldColor, newColor, rows, cols);
        FloodFill2(image, row, col - 1, oldColor, newColor, rows, cols);
    }

    public static ListNode MiddleNode(ListNode head)
    {
        //O(n) O(1)
        //var length = GetLength(head);
        //var mid = length / 2 + length % 2;
        //for (int i = 0; i < length - mid; i++)
        //{
        //    head = head.next;
        //}

        //return head;

        //better solution O(n) O(1)
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null)
        {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }
    private int GetLength(ListNode node)
    {
        var length = 0;
        while (node != null)
        {
            length++;
            node = node.next;
        }
        return length;
    }

    public static int LengthOfLongestSubstring(string s)
    {
        int maxLen = 0;
        int left = 0;
        var map = new Dictionary<char, int>();

        for (int right = 0; right < s.Length; right++)
        {
            char c = s[right];

            if (map.ContainsKey(c))

            {
                // Move left only forward to avoid shrinking backward
                left = Math.Max(map[c] + 1, left);
            }
            map[c] = right; // Update last seen index
            maxLen = Math.Max(maxLen, right - left + 1);
        }

        return maxLen;
    }

    public static int LengthOfLongestSubstring2(string s)
    {
        var text = s.ToCharArray();
        var set = new HashSet<char>();
        int left = 0, right = 0, max = 0;
        while (right < text.Length)
        {
            if (!set.Contains(text[right]))
            {
                set.Add(text[right]);
                right++;
                max = Math.Max(max, right - left);
            }
            else
            {
                set.Remove(text[left]);
                left++;
            }
        }
        return max;
    }

    public static string LongestPalindromeSubstring(string s)
    {
        //O(N2) / O(1)
        // Check if the input string is empty, return an empty string if so
        if (string.IsNullOrEmpty(s))
            return "";

        // Initialize variables to store the indices of the longest palindrome found
        int[] longestPalindromeIndices = { 0, 0 };

        // Loop through the characters in the input string
        for (int i = 0; i < s.Length; ++i)
        {
            // Find the indices of the longest palindrome centered at character i
            int[] currentIndices = ExpandAroundCenter(s, i, i);

            // Compare the length of the current palindrome with the longest found so far
            if (currentIndices[1] - currentIndices[0] > longestPalindromeIndices[1] - longestPalindromeIndices[0])
            {
                // Update the longest palindrome indices if the current one is longer
                longestPalindromeIndices = currentIndices;
            }

            // Check if there is a possibility of an even-length palindrome centered at
            // character i and i+1
            if (i + 1 < s.Length && s[i] == s[i + 1])
            {
                // Find the indices of the longest even-length palindrome centered at characters
                // i and i+1
                int[] evenIndices = ExpandAroundCenter(s, i, i + 1);

                // Compare the length of the even-length palindrome with the longest found so
                // far
                if (evenIndices[1] - evenIndices[0] > longestPalindromeIndices[1] - longestPalindromeIndices[0])
                {
                    // Update the longest palindrome indices if the even-length one is longer
                    longestPalindromeIndices = evenIndices;
                }
            }
        }

        // Extract and return the longest palindrome substring using the indices
        return s.Substring(longestPalindromeIndices[0], longestPalindromeIndices[1] - longestPalindromeIndices[0] + 1);
    }
    // Helper function to find and return the indices of the longest palindrome
    // extended from s[i..j] (inclusive) by expanding around the center
    private static int[] ExpandAroundCenter(string s, int i, int j)
    {
        // Expand the palindrome by moving outward from the center while the characters match
        while (i >= 0 && j < s.Length && s[i] == s[j])
        {
            i--; // Move the left index to the left
            j++; // Move the right index to the right
        }
        // Return the indices of the longest palindrome found
        return new int[] { i + 1, j - 1 };
    }

    public static string LongestPalindromeSubstring2(string s)
    {
        //Manacher's Algorithm
        string T = "^#" + string.Join("#", s.ToCharArray()) + "#$";
        int n = T.Length;
        int[] P = new int[n];
        int C = 0, R = 0;

        for (int i = 1; i < n - 1; i++)
        {
            P[i] = (R > i) ? Math.Min(R - i, P[2 * C - i]) : 0;
            while (T[i + 1 + P[i]] == T[i - 1 - P[i]])
                P[i]++;

            if (i + P[i] > R)
            {
                C = i;
                R = i + P[i];
            }
        }

        int max_len = P.Max();
        int center_index = Array.IndexOf(P, max_len);
        return s.Substring((center_index - max_len) / 2, max_len);
    }

    public static string LongestPalindromeSubstring3(string s)
    {
        // O(N) / O(N)
        // Step 1: Preprocess the input string
        StringBuilder processedStr = new StringBuilder("^#");
        foreach (char c in s)
        {
            processedStr.Append(c).Append("#");
        }
        processedStr.Append("$");
        string modifiedString = processedStr.ToString();

        // Step 2: Initialize variables for the algorithm
        int strLength = modifiedString.Length;
        int[] palindromeLengths = new int[strLength];
        int center = 0;  // Current center of the palindrome
        int rightEdge = 0;  // Rightmost edge of the palindrome

        // Step 3: Loop through the modified string to find palindromes
        for (int i = 1; i < strLength - 1; i++)
        {
            //- # a # b # a # b # a # +
            //0 1 2 3 4 5 6 7 8 9 0 1 2
            if (rightEdge > i)
            {
                palindromeLengths[i] = Math.Min(rightEdge - i, palindromeLengths[2 * center - i]);
            }
            else
            {
                palindromeLengths[i] = 0;
            }

            var l = modifiedString[i + 1 + palindromeLengths[i]];
            var r = modifiedString[i - 1 - palindromeLengths[i]];
            // Expand the palindrome around the current character
            while (modifiedString[i + 1 + palindromeLengths[i]] == modifiedString[i - 1 - palindromeLengths[i]])
            {
                palindromeLengths[i]++;
            }

            // Update the rightmost edge and center if necessary
            if (i + palindromeLengths[i] > rightEdge)
            {
                center = i;
                rightEdge = i + palindromeLengths[i];
            }
        }

        // Step 4: Find the longest palindrome and its center
        int maxLength = 0;
        int maxCenter = 0;
        for (int i = 0; i < strLength; i++)
        {
            if (palindromeLengths[i] > maxLength)
            {
                maxLength = palindromeLengths[i];
                maxCenter = i;
            }
        }

        // Step 5: Extract the longest palindrome from the modified string
        int start = (maxCenter - maxLength) / 2;
        int end = start + maxLength;

        // Return the longest palindrome in the original string
        return s.Substring(start, end - start);
    }

    public static int MyAtoi(string s)
    {
        s = s.Trim();
        if (string.IsNullOrEmpty(s))
        {
            return 0;
        }

        int i = 0, num = 0, sign = 1;

        if (s[0] == '+' || s[0] == '-')
        {
            sign = (s[0] == '-') ? -1 : 1;
            i++;
        }

        while (i < s.Length && Char.IsDigit(s[i]))
        {
            // 0 -> 48
            // 1 -> 49
            // 4 -> 52
            int digit = s[i] - '0';
            if (num > (Int32.MaxValue - digit) / 10)
            {
                return (sign == 1) ? Int32.MaxValue : Int32.MinValue;
            }
            num = num * 10 + digit;
            i++;
        }

        return sign * num;
    }

    // Container with most water
    public static int MaxArea(int[] height)
    {
        int left = 0;
        int right = height.Length - 1;
        int maxArea = 0;

        while (left < right)
        {
            int width = right - left;
            int minHeight = Math.Min(height[left], height[right]);
            int area = width * minHeight;
            maxArea = Math.Max(maxArea, area);

            if (height[left] < height[right])
            {
                left++;
            }
            else
            {
                right--;
            }
        }

        return maxArea;
    }

    public static IList<IList<int>> ThreeSum(int[] nums)
    {
        Array.Sort(nums);
        IList<IList<int>> result = new List<IList<int>>();

        for (int i = 0; i < nums.Length; i++)
        {
            if (i > 0 && nums[i] == nums[i - 1])
            {
                continue; // Skip duplicate values for i
            }

            int j = i + 1, k = nums.Length - 1;

            while (j < k)
            {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum == 0)
                {
                    result.Add(new List<int> { nums[i], nums[j], nums[k] });
                    j++;
                    k--;
                    while (j < k && nums[j] == nums[j - 1]) j++;
                    while (j < k && nums[k] == nums[k + 1]) k--;
                }
                else if (sum < 0)
                {
                    j++;
                }
                else
                {
                    k--;
                }
            }
        }

        return result;
    }

    public static IList<string> LetterCombinations(string digits)
    {
        if (digits.Length == 0) return new List<string>();
        string[] comb = ["", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"];
        List<string> result = [];
        Solve(0, digits, comb, result, "");
        return result;
    }
    private static void Solve(int index, string digits, string[] comb, List<string> result, string temp)
    {
        if (index == digits.Length)
        {
            result.Add(temp);
            return;
        }

        foreach (char ch in comb[digits[index] - '0'])
        {
            Solve(index + 1, digits, comb, result, temp + ch);
        }
    }

    public static int SearchInRotatedSortedArray(int[] nums, int target)
    {
        int len = nums.Length - 1;
        int start = 0;
        int end = len;

        while (start <= end)
        {
            int mid = (start + end) / 2;

            if (target == nums[mid])
            {
                return mid;
            }

            if (nums[mid] < nums[end])
            {
                if (target > nums[mid] && target <= nums[end])
                {
                    start = mid + 1;
                }
                else
                {
                    end = mid - 1;
                }
            }
            else
            {
                if (target >= nums[start] && target < nums[mid])
                {
                    end = mid - 1;

                }
                else
                {
                    start = mid + 1;
                }
            }
        }

        return -1;
    }


    public static IList<IList<int>> CombinationSum(int[] candidates, int target)
    {
        List<IList<int>> result = [];
        Array.Sort(candidates);
        Backtrack(candidates, target, 0, [], result);
        return result;
    }
    private static void Backtrack(int[] candidates, int remain, int start, List<int> current, List<IList<int>> result)
    {
        if (remain == 0)
        {
            result.Add([.. current]);
            return;
        }
        if (remain < 0) return;

        for (int i = start; i < candidates.Length; i++)
        {
            if (candidates[i] > remain)
            {
                break;
            }

            current.Add(candidates[i]);
            Backtrack(candidates, remain - candidates[i], i, current, result);
            current.RemoveAt(current.Count - 1);
        }
    }

    public static IList<IList<int>> Permute(int[] nums)
    {
        List<IList<int>> result = [];
        BacktrackPermute(nums, new List<int>(), result);
        return result;
    }
    private static void BacktrackPermute(int[] nums, List<int> path, IList<IList<int>> result)
    {
        if (path.Count == nums.Length)
        {
            result.Add([.. path]);
            return;
        }

        foreach (int num in nums)
        {
            if (path.Contains(num))
                continue;

            path.Add(num);
            BacktrackPermute(nums, path, result);
            path.RemoveAt(path.Count - 1);
        }
    }







    //Booking
    public static int RemoveDuplicates(int[] nums)
    {
        if (nums.Length == 0) return 0;

        int slow = 0;
        for (int fast = 1; fast < nums.Length; fast++)
        {
            if (nums[fast] != nums[slow])
            {
                slow++;
                nums[slow] = nums[fast];
            }
        }

        return slow + 1;
    }

    public static int FindKthLargest(int[] nums, int k)
    {
        // Min-heap to store k largest elements
        var minHeap = new PriorityQueue<int, int>();

        foreach (int num in nums)
        {
            minHeap.Enqueue(num, num); // Add element to heap

            if (minHeap.Count > k)
            {
                minHeap.Dequeue(); // Remove smallest if size exceeds k
            }
        }

        // Root of heap is the k-th largest
        return minHeap.Peek();
    }

    public static int[] FindKLargest(int[] nums, int k)
    {
        var minHeap = new PriorityQueue<int, int>();

        foreach (int num in nums)
        {
            if (minHeap.Count < k)
            {
                minHeap.Enqueue(num, num);
            }
            else if (num > minHeap.Peek())
            {
                minHeap.Dequeue();
                minHeap.Enqueue(num, num);
            }
        }

        var result = new int[k];
        for (int i = k - 1; i >= 0; i--)
        {
            result[i] = minHeap.Dequeue();
        }

        return result;
    }

    public static List<int> awardTopKHotels(
        string positiveKeywords,
        string negativeKeywords,
        List<int> hotelIds,
        List<string> reviews,
        int k)
    {
        var pSet = new HashSet<string>(GetSanitisedList(positiveKeywords));
        var nSet = new HashSet<string>(GetSanitisedList(negativeKeywords));

        var result = new Dictionary<int, int>();

        for (int i = 0; i < hotelIds.Count; i++)
        {
            var reviewWords = GetSanitisedList(reviews[i]);

            var score = 0;

            foreach (var word in reviewWords)
            {
                if (pSet.Contains(word))
                    score += 3;
                else if (nSet.Contains(word))
                    score -= 1;
            }

            if (result.ContainsKey(hotelIds[i]))
                result[hotelIds[i]] += score;
            else
                result[hotelIds[i]] = score;
        }

        return result
            .OrderByDescending(x => x.Value)
            .ThenBy(x => x.Key)
            .Take(k)
            .Select(x => x.Key)
            .ToList();
    }

    private static string[] GetSanitisedList(string text)
    {
        return text
            .ToLower()
            .Replace(",", "")
            .Replace(".", "")
            .Replace("!", "")
            .Split(' ', StringSplitOptions.RemoveEmptyEntries);
    }




}