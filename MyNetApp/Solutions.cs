namespace MyNetApp;

public class Solutions
{
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

    public static int LengthOfLongestSubstring(string s)
    {
        int n = s.Length;
        int maxLen = 0;
        int left = 0;
        var map = new Dictionary<char, int>();

        for (int right = 0; right < n; right++)
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
        var tab = new int[n + 1];
        if (tab.Length > 0) tab[0] = 1;
        if (tab.Length > 1) tab[1] = 1;
        for (var i = 2; i < tab.Length; i++)
            tab[i] = tab[i - 1] + tab[i - 2];
        return tab[n];
    }

    public static int MaxDepth(TreeNode root)
    {
        if (root == null)
            return 0;

        var val = 1 + Math.Max(MaxDepth(root.left), MaxDepth(root.right));

        return val;
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
        return version >= 9;
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

    public static int MaxProfit(int[] prices)
    {
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

        int minPrice = int.MaxValue;
        int maxProfit = 0;

        foreach (int currentPrice in prices)
        {
            minPrice = Math.Min(currentPrice, minPrice);
            maxProfit = Math.Max(maxProfit, currentPrice - minPrice);
        }

        return maxProfit;
    }

    public static bool HasCycle(ListNode head)
    {
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

        ListNode resultNode = null;
        while (head != null)
        {
            resultNode = new ListNode(head.val, resultNode);
            head = head.next;
        }
        return resultNode;
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

    public static int TreeHeight(TreeNode root)
    {
        if (root == null)
        {
            return 0;
        }

        return int.Max(TreeHeight(root.left), TreeHeight(root.right)) + 1;
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
    public static void FloodFill2(int[][] image, int row, int col, int oldColor, int newColor, int rows, int cols)
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

}