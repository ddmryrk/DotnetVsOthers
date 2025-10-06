using System;
using System.Collections.Generic;
using System.Linq;

public class LeetCodeAlgorithms
{
    // ==================== SLIDING WINDOW ====================
    
    // Fixed Size Sliding Window
    public int MaxSumSubarray(int[] nums, int k)
    {
        int maxSum = 0, windowSum = 0;
        
        // Calculate sum of first window
        for (int i = 0; i < k; i++)
            windowSum += nums[i];
        
        maxSum = windowSum;
        
        // Slide the window
        for (int i = k; i < nums.Length; i++)
        {
            windowSum = windowSum - nums[i - k] + nums[i];
            maxSum = Math.Max(maxSum, windowSum);
        }
        
        return maxSum;
    }
    
    // Variable Size Sliding Window
    public int LengthOfLongestSubstring(string s)
    {
        var seen = new HashSet<char>();
        int left = 0, maxLen = 0;
        
        for (int right = 0; right < s.Length; right++)
        {
            while (seen.Contains(s[right]))
            {
                seen.Remove(s[left]);
                left++;
            }
            seen.Add(s[right]);
            maxLen = Math.Max(maxLen, right - left + 1);
        }
        
        return maxLen;
    }
    
    // ==================== TWO POINTERS ====================
    
    // Two Sum in Sorted Array
    public int[] TwoSum(int[] nums, int target)
    {
        int left = 0, right = nums.Length - 1;
        
        while (left < right)
        {
            int sum = nums[left] + nums[right];
            if (sum == target)
                return new int[] { left, right };
            else if (sum < target)
                left++;
            else
                right--;
        }
        
        return new int[] { -1, -1 };
    }
    
    // Remove Duplicates
    public int RemoveDuplicates(int[] nums)
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
    
    // ==================== BINARY SEARCH ====================
    
    public int BinarySearch(int[] nums, int target)
    {
        int left = 0, right = nums.Length - 1;
        
        while (left <= right)
        {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] == target)
                return mid;
            else if (nums[mid] < target)
                left = mid + 1;
            else
                right = mid - 1;
        }
        
        return -1;
    }
    
    // Find First/Last Position
    public int[] SearchRange(int[] nums, int target)
    {
        int first = FindFirst(nums, target);
        int last = FindLast(nums, target);
        return new int[] { first, last };
    }
    
    private int FindFirst(int[] nums, int target)
    {
        int left = 0, right = nums.Length - 1, result = -1;
        
        while (left <= right)
        {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] == target)
            {
                result = mid;
                right = mid - 1; // Continue searching left
            }
            else if (nums[mid] < target)
                left = mid + 1;
            else
                right = mid - 1;
        }
        
        return result;
    }
    
    private int FindLast(int[] nums, int target)
    {
        int left = 0, right = nums.Length - 1, result = -1;
        
        while (left <= right)
        {
            int mid = left + (right - left) / 2;
            
            if (nums[mid] == target)
            {
                result = mid;
                left = mid + 1; // Continue searching right
            }
            else if (nums[mid] < target)
                left = mid + 1;
            else
                right = mid - 1;
        }
        
        return result;
    }
    
    // ==================== DEPTH-FIRST SEARCH (DFS) ====================
    
    // Tree Node Definition
    public class TreeNode
    {
        public int val;
        public TreeNode left;
        public TreeNode right;
        public TreeNode(int val = 0, TreeNode left = null, TreeNode right = null)
        {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }
    
    // DFS - Recursive
    public List<int> InorderTraversal(TreeNode root)
    {
        var result = new List<int>();
        InorderHelper(root, result);
        return result;
    }
    
    private void InorderHelper(TreeNode node, List<int> result)
    {
        if (node == null) return;
        
        InorderHelper(node.left, result);
        result.Add(node.val);
        InorderHelper(node.right, result);
    }
    
    // DFS - Iterative with Stack
    public List<int> InorderTraversalIterative(TreeNode root)
    {
        var result = new List<int>();
        var stack = new Stack<TreeNode>();
        TreeNode curr = root;
        
        while (curr != null || stack.Count > 0)
        {
            while (curr != null)
            {
                stack.Push(curr);
                curr = curr.left;
            }
            
            curr = stack.Pop();
            result.Add(curr.val);
            curr = curr.right;
        }
        
        return result;
    }
    
    // DFS on Graph
    public bool HasPath(Dictionary<int, List<int>> graph, int start, int end, HashSet<int> visited = null)
    {
        if (visited == null) visited = new HashSet<int>();
        if (start == end) return true;
        if (visited.Contains(start)) return false;
        
        visited.Add(start);
        
        if (graph.ContainsKey(start))
        {
            foreach (int neighbor in graph[start])
            {
                if (HasPath(graph, neighbor, end, visited))
                    return true;
            }
        }
        
        return false;
    }
    
    // ==================== BREADTH-FIRST SEARCH (BFS) ====================
    
    // BFS on Tree
    public List<List<int>> LevelOrder(TreeNode root)
    {
        var result = new List<List<int>>();
        if (root == null) return result;
        
        var queue = new Queue<TreeNode>();
        queue.Enqueue(root);
        
        while (queue.Count > 0)
        {
            int levelSize = queue.Count;
            var currentLevel = new List<int>();
            
            for (int i = 0; i < levelSize; i++)
            {
                TreeNode node = queue.Dequeue();
                currentLevel.Add(node.val);
                
                if (node.left != null) queue.Enqueue(node.left);
                if (node.right != null) queue.Enqueue(node.right);
            }
            
            result.Add(currentLevel);
        }
        
        return result;
    }
    
    // BFS on Grid (Shortest Path)
    public int ShortestPath(int[][] grid, int[] start, int[] end)
    {
        if (grid == null || grid.Length == 0) return -1;
        
        int rows = grid.Length, cols = grid[0].Length;
        var queue = new Queue<(int row, int col, int dist)>();
        var visited = new bool[rows, cols];
        var directions = new int[][] { new int[] { 0, 1 }, new int[] { 1, 0 }, new int[] { 0, -1 }, new int[] { -1, 0 } };
        
        queue.Enqueue((start[0], start[1], 0));
        visited[start[0], start[1]] = true;
        
        while (queue.Count > 0)
        {
            var (row, col, dist) = queue.Dequeue();
            
            if (row == end[0] && col == end[1])
                return dist;
            
            foreach (var dir in directions)
            {
                int newRow = row + dir[0];
                int newCol = col + dir[1];
                
                if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols && 
                    !visited[newRow, newCol] && grid[newRow][newCol] == 0)
                {
                    visited[newRow, newCol] = true;
                    queue.Enqueue((newRow, newCol, dist + 1));
                }
            }
        }
        
        return -1;
    }
    
    // ==================== DYNAMIC PROGRAMMING ====================
    
    // 1D DP - Fibonacci
    public int Fibonacci(int n)
    {
        if (n <= 1) return n;
        
        var dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;
        
        for (int i = 2; i <= n; i++)
        {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        
        return dp[n];
    }
    
    // Space Optimized DP
    public int FibonacciOptimized(int n)
    {
        if (n <= 1) return n;
        
        int prev2 = 0, prev1 = 1;
        
        for (int i = 2; i <= n; i++)
        {
            int curr = prev1 + prev2;
            prev2 = prev1;
            prev1 = curr;
        }
        
        return prev1;
    }
    
    // 2D DP - Unique Paths
    public int UniquePaths(int m, int n)
    {
        var dp = new int[m, n];
        
        // Initialize first row and column
        for (int i = 0; i < m; i++) dp[i, 0] = 1;
        for (int j = 0; j < n; j++) dp[0, j] = 1;
        
        for (int i = 1; i < m; i++)
        {
            for (int j = 1; j < n; j++)
            {
                dp[i, j] = dp[i - 1, j] + dp[i, j - 1];
            }
        }
        
        return dp[m - 1, n - 1];
    }
    
    // ==================== BACKTRACKING ====================
    
    // Generate All Permutations
    public List<List<int>> Permute(int[] nums)
    {
        var result = new List<List<int>>();
        var current = new List<int>();
        var used = new bool[nums.Length];
        
        BacktrackPermute(nums, current, used, result);
        return result;
    }
    
    private void BacktrackPermute(int[] nums, List<int> current, bool[] used, List<List<int>> result)
    {
        if (current.Count == nums.Length)
        {
            result.Add(new List<int>(current));
            return;
        }
        
        for (int i = 0; i < nums.Length; i++)
        {
            if (used[i]) continue;
            
            current.Add(nums[i]);
            used[i] = true;
            
            BacktrackPermute(nums, current, used, result);
            
            current.RemoveAt(current.Count - 1);
            used[i] = false;
        }
    }
    
    // Generate All Subsets
    public List<List<int>> Subsets(int[] nums)
    {
        var result = new List<List<int>>();
        var current = new List<int>();
        
        BacktrackSubsets(nums, 0, current, result);
        return result;
    }
    
    private void BacktrackSubsets(int[] nums, int start, List<int> current, List<List<int>> result)
    {
        result.Add(new List<int>(current));
        
        for (int i = start; i < nums.Length; i++)
        {
            current.Add(nums[i]);
            BacktrackSubsets(nums, i + 1, current, result);
            current.RemoveAt(current.Count - 1);
        }
    }
    
    // ==================== TRIE (PREFIX TREE) ====================
    
    public class TrieNode
    {
        public TrieNode[] children;
        public bool isEndOfWord;
        
        public TrieNode()
        {
            children = new TrieNode[26];
            isEndOfWord = false;
        }
    }
    
    public class Trie
    {
        private TrieNode root;
        
        public Trie()
        {
            root = new TrieNode();
        }
        
        public void Insert(string word)
        {
            TrieNode node = root;
            foreach (char c in word)
            {
                int index = c - 'a';
                if (node.children[index] == null)
                    node.children[index] = new TrieNode();
                node = node.children[index];
            }
            node.isEndOfWord = true;
        }
        
        public bool Search(string word)
        {
            TrieNode node = SearchHelper(word);
            return node != null && node.isEndOfWord;
        }
        
        public bool StartsWith(string prefix)
        {
            return SearchHelper(prefix) != null;
        }
        
        private TrieNode SearchHelper(string word)
        {
            TrieNode node = root;
            foreach (char c in word)
            {
                int index = c - 'a';
                if (node.children[index] == null)
                    return null;
                node = node.children[index];
            }
            return node;
        }
    }
    
    // ==================== UNION FIND (DISJOINT SET) ====================
    
    public class UnionFind
    {
        private int[] parent;
        private int[] rank;
        
        public UnionFind(int n)
        {
            parent = new int[n];
            rank = new int[n];
            
            for (int i = 0; i < n; i++)
            {
                parent[i] = i;
                rank[i] = 0;
            }
        }
        
        public int Find(int x)
        {
            if (parent[x] != x)
                parent[x] = Find(parent[x]); // Path compression
            return parent[x];
        }
        
        public bool Union(int x, int y)
        {
            int rootX = Find(x);
            int rootY = Find(y);
            
            if (rootX == rootY) return false;
            
            // Union by rank
            if (rank[rootX] < rank[rootY])
                parent[rootX] = rootY;
            else if (rank[rootX] > rank[rootY])
                parent[rootY] = rootX;
            else
            {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
            
            return true;
        }
        
        public bool Connected(int x, int y)
        {
            return Find(x) == Find(y);
        }
    }
    
    // ==================== MONOTONIC STACK ====================
    
    // Next Greater Element
    public int[] NextGreaterElement(int[] nums)
    {
        var result = new int[nums.Length];
        var stack = new Stack<int>(); // Store indices
        
        Array.Fill(result, -1);
        
        for (int i = 0; i < nums.Length; i++)
        {
            while (stack.Count > 0 && nums[stack.Peek()] < nums[i])
            {
                int index = stack.Pop();
                result[index] = nums[i];
            }
            stack.Push(i);
        }
        
        return result;
    }
    
    // ==================== HEAP OPERATIONS ====================
    
    // Find K Largest Elements
    public int[] FindKLargest(int[] nums, int k)
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
}

// ==================== USAGE EXAMPLES ====================
class Program
{
    static void Main()
    {
        var algo = new LeetCodeAlgorithms();
        
        // Example usage
        int[] nums = { 1, 2, 3, 4, 5 };
        Console.WriteLine(algo.MaxSumSubarray(nums, 3)); // Sliding window
        
        int[] sorted = { 1, 2, 3, 4, 5, 6 };
        var twoSumResult = algo.TwoSum(sorted, 9);
        Console.WriteLine($"Two sum indices: [{twoSumResult[0]}, {twoSumResult[1]}]");
        
        // Binary search
        Console.WriteLine(algo.BinarySearch(sorted, 4));
        
        // Trie usage
        var trie = new LeetCodeAlgorithms.Trie();
        trie.Insert("apple");
        Console.WriteLine(trie.Search("apple"));   // True
        Console.WriteLine(trie.StartsWith("app")); // True
        
        // Union Find
        var uf = new LeetCodeAlgorithms.UnionFind(5);
        uf.Union(0, 1);
        uf.Union(2, 3);
        Console.WriteLine(uf.Connected(0, 1)); // True
        Console.WriteLine(uf.Connected(0, 2)); // False
    }
}