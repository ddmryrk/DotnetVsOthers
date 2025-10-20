
import java.util.Arrays;
import java.util.List;

public class Main {

    public static void main(String[] args) {
        var solutions = new Solutions();
        int[][] arr = {{1, 3}, {6, 9}};
        //solutions.insert(arr, new int[]{2,5});
        //solutions.uniquePaths2(4,4);
        //solutions.sortColors(new int[]{2,0,2,1,1,0});
        //solutions.subsets(new int[]{1,2,3});
        List<List<String>> a = Arrays.asList(
                Arrays.asList("John", "a", "b", "d"),
                Arrays.asList("John", "a", "c"),
                Arrays.asList("Mary", "m"),
                Arrays.asList("John", "e", "f")
        //John a b c d
        //John e
        //Marry m
        );
        //solutions.accountsMerge2(a);
        //solutions.coinChange(new int[]{1,2,5}, 11);
        TreeNode t = new TreeNode(3, new TreeNode(9), new TreeNode(20, new TreeNode(15), new TreeNode(7)));
        //solutions.levelOrder(t);

        var lru = new LRUCache(2);
        lru.put(1, 1);
        lru.put(2,2);
        lru.get(1);
        lru.put(3,3);
        lru.get(2);
        lru.put(4,4);
        lru.get(1);
        lru.get(3);
        lru.get(4);
        


        /*
        int[] result = solutions.twoSum(new int[] { 3, 2, 4 }, 6);
        
        System.out.println("Indices: " + result[0] + ", " + result[1]);

        System.out.println("Is Valid: " + solutions.validParentheses("()[]{}"));

        System.out.println("Merged: " + solutions.mergeTwoSortedLists(
                new ListNode(1, new ListNode(2, new ListNode(4))),
                new ListNode(1, new ListNode(3, new ListNode(4)))));

        System.out.println("Sum: " + solutions.addBinary("10", "101"));

        System.out.println("Climb Stairs: " + solutions.climbStairs(11));

        System.out.println("Max Depth: " + solutions.maxDepth(
                new TreeNode(3, new TreeNode(9), new TreeNode(20, new TreeNode(15), new TreeNode(7)))
        ));
         */
    }
}
