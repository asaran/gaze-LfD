# Understanding Teacher Gaze Patterns for Robot Learning

This respository contains code for the following [paper](https://arxiv.org/pdf/1907.07202v4.pdf):
> A. Saran*, E.S. Short, A. Thomaz, S. Niekum.
Understanding Teacher Gaze Patterns for Robot Learning.
Conference on Robot Learning (CoRL), November 2019. 

The data for the user study associated with this paper is not permissible for release under IRB constraints. We include dummy data with our code wherever possible.

## Code for Statistical Analysis of Gaze Patterns (stat_analysis/) 
### Pouring Task 

   * `stat_analysis/pouring/exp1.py`: Percentage of time during entire demo spent fixating on objects, gripper or other parts of workspace 
   * `stat_analysis/pouring/exp2.py`: Perecentage accuarcy to predict reference frame per keyframe
   * `stat_analysis/pouring/exp3.py`: Gaze Patterns between Novice and Expert Users
   * `stat_analysis/pouring/exp4.py`: Gaze-patterns around step KF and non-step KF (KT demos only)
   * `stat_analysis/pouring/gaze_filtering_demo.py`: A demo file with dummy data for gaze fixation filtering

### Placement Task 
   * `stat_analysis/placement/exp1.py`: Percentage of time during entire demo spent fixating on objects, gripper or other parts of workspace 
   * `stat_analysis/placement/exp2.py`: Gaze fixations on bowl and plate for different relative placement strategy

## Code for SubTask Classification (subtask_detection/)
   1. Non-local neural network
   2. Compact Generalized Non-local neural network 
   * Training data for these networks cannot be made available due to IRB restrictions, however we do link trained models and some dummy data for validation.
   * Detailed instructions to run the code are available in `subtask_detection/README.md`.
   * The files which have been modified to add gaze data into the training framework are train_val.py and models/resnet.py.

## Code for Reward Learning with Gaze-augmented BIRL (gaze_birl/)

* To run Gaze-augmented BIRL for KT demos:
	* instruction relative to plate: `./test_gaze_birl.py --demo_type 'KT' --exp 'plate' --use_gaze `
	* instruction relative to bowl: `./test_gaze_birl.py --demo_type 'KT' --exp 'bowl' --use_gaze `

* To run Gaze-augmented BIRL for video demos:
	* instruction relative to plate: `./test_gaze_birl.py --demo_type 'video' --exp 'plate' --use_gaze `
	* instruction relative to bowl: `./test_gaze_birl.py --demo_type 'video' --exp 'bowl' --use_gaze `

* To run standard BIRL for KT demos:
	* instruction relative to plate: `./test_gaze_birl.py --demo_type 'KT' --exp 'plate' `
	* instruction relative to bowl: `./test_gaze_birl.py --demo_type 'KT' --exp 'bowl' `

* To run standard BIRL for video demos:
	* instruction relative to plate: `./test_gaze_birl.py --demo_type 'video' --exp 'plate' `
	* instruction relative to bowl: `./test_gaze_birl.py --demo_type 'video' --exp 'bowl' `


## Bibliography
If you find our work to be useful in your research, please cite:
```
@article{saran2019understanding,
  title={Understanding Teacher Gaze Patterns for Robot Learning},
  author={Saran, Akanksha and Short, Elaine Schaertl and Thomaz, Andrea and Niekum, Scott},
  booktitle={Conference on Robot Learning},
  year={2019},
  organization={PMLR}
}
```
