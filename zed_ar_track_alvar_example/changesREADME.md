# Changes in the script and RViz
In order to work correctly we changed the parameters of camera_model = zed2

marker_size = 20

max_new_marker_error = 0.2

Also in RViz we changed "Fixed frame map" to "Fixed frame base_link" and enable Pose

## To launch it
``` cd catkin_ws/src/zed-implementations ```

``` roslaunch zed_ar_track_alvar_example zed_ar_track_alvar.launch ```
