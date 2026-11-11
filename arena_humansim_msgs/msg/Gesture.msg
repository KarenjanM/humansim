# One active attention channel of an agent
string slot                  # head | arm | arm_l | arm_r | halt | halt_l | halt_r | body
geometry_msgs/Point at       # target, world frame (aimed slots)
string clip                  # body: animation clip name
string hand                  # arm/halt: dominant hand l | r, empty = default
bool render_pose_override    # body: `at` is a render-only position override (xy), not a gaze/point target
bool grip                    # receive: close the selected hand, false keeps it open
