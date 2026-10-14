# State of a single pedestrian agent
uint8 KIND_HUMAN=0
uint8 KIND_ROBOT=1

int32 agent_id
string name                  # empty = unnamed
geometry_msgs/Pose2D pose
geometry_msgs/Vector3 velocity       # linear velocity (x, y, z=0)
float64 desired_velocity              # preferred walking speed (m/s)
float64 radius                        # collision radius (m)

# Per-agent parameters (0.0 / empty = use global default)
float64 vision_range         # 0.0 = use global default
float64 vision_fov           # 0.0 = use global default
float64 relaxation_time      # 0.0 = use global default
float64 repulsion_strength   # 0.0 = use global default
float64 repulsion_range      # 0.0 = use global default
string agent_type            # empty = use global default

arena_humansim_msgs/Waypoints waypoints

uint8 kind 0                 # KIND_HUMAN | KIND_ROBOT
uint8 animation_state 0
string policy                # registered local planner name; empty = teleport-only
string policy_params         # opaque JSON blob for planner

string handedness            # l | r, empty = right
Gesture[] gestures           # active attention channels

int32 interaction_id -1      # active interaction, -1 = none
uint8 interaction_type       # InteractionType, valid when interaction_id >= 0
