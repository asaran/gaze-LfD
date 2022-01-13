# Code for Gaze-augmented BIRL.

Since the data for th user study cannot be released under IRB restrictions, we provide a dummy text file to load the data for gaze-augmented BIRl. Normalized object locations for 5 demonstrations are loaded from `objects.txt`. 

- To run Gaze-augmented BIRL for video demos:

	* instruction relative to plate   ```./test_gaze_birl.py --demo_type 'video' --exp 'plate' --use_gaze ```
	* instruction relative to bowl	```./test_gaze_birl.py --demo_type 'video' --exp 'bowl' --use_gaze ```

- To run Gaze-augmented BIRL for KT demos:

	* instruction relative to plate   ```./test_gaze_birl.py --demo_type 'KT' --exp 'plate' --use_gaze ```
	* instruction relative to bowl	```./test_gaze_birl.py --demo_type 'KT' --exp 'bowl' --use_gaze ```

- To run standard BIRL for video demos:

	* instruction relative to plate   ```./test_gaze_birl.py --demo_type 'video' --exp 'plate'  ```
	* instruction relative to bowl	```./test_gaze_birl.py --demo_type 'video' --exp 'bowl'  ```

- To run standard BIRL for KT demos:

	* instruction relative to plate   ```./test_gaze_birl.py --demo_type 'KT' --exp 'plate'  ```
	* instruction relative to bowl	```./test_gaze_birl.py --demo_type 'KT' --exp 'bowl'  ```
