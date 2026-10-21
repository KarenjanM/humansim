# One elevator cabin pedestrians ride, engine frame
string name                          # unique cabin name
string destination                   # cabin a ride arrives in, empty = none
geometry_msgs/Pose2D boarding        # where a rider holds inside the cabin
geometry_msgs/Pose2D doorway         # outer edge of the door, facing the cabin
geometry_msgs/Pose2D queue           # where a rider waits, clear of the door threshold
geometry_msgs/Point32[] footprint    # cabin polygon, z unused
bool door_open                       # door state when added
