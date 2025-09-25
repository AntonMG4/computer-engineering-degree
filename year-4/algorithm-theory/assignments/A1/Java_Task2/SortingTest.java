import java.util.Random;

public class SortingTest {
    
    public static long measureTime(int[] arr, String sortType) {
        long startTime = System.nanoTime();
        for(int i=0;i<3;i++){
            switch (sortType) {
                case "quickSort":
                    SortingAlgorithms.quickSort(arr, 0, arr.length - 1);
                    break;
                case "insertionSort":
                    SortingAlgorithms.insertionSort(arr);
                    break;
                case "heapSort":
                    SortingAlgorithms.heapSort(arr);
                    break;
                case "mergeSort":
                    SortingAlgorithms.mergeSort(arr, 0, arr.length - 1);
                    break;
            }
        }
        long endTime = System.nanoTime();
        return (endTime - startTime)/3;
    }

    public static int[] generateRandomArray(int size) {
        int[] arr = new int[size];
        Random rand = new Random();
        for (int i = 0; i < size; i++) {
            arr[i] = rand.nextInt(10000);
        }
        return arr;
    }

    public static void main(String[] args) {
        int[] sizes = {1000,2500, 5000, 7500, 10000};  // Puedes cambiar los tamaños

        for (int size : sizes) {
            int[] arr = generateRandomArray(size);
            System.out.println(("Size: "+ size));

            // QuickSort
            int[] arrCopy = arr.clone();
            long timeTaken = measureTime(arrCopy, "quickSort");
            System.out.println("QuickSort (size=" + size + "): " + timeTaken / 1_000_000.0 + " ms");
            
            // InsertionSort
            arrCopy = arr.clone();
            timeTaken = measureTime(arrCopy, "insertionSort");
            System.out.println("InsertionSort (size=" + size + "): " + timeTaken / 1_000_000.0 + " ms");
            
            // HeapSort
            arrCopy = arr.clone();
            timeTaken = measureTime(arrCopy, "heapSort");
            System.out.println("HeapSort (size=" + size + "): " + timeTaken / 1_000_000.0 + " ms");
            
            // MergeSort
            arrCopy = arr.clone();
            timeTaken = measureTime(arrCopy, "mergeSort");
            System.out.println("MergeSort (size=" + size + "): " + timeTaken / 1_000_000.0 + " ms");
        }
    }
}
