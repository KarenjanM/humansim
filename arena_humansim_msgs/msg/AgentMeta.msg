# Slow-changing attributes of the engine's own agents, latched, sent on change
std_msgs/Header header
string[] policies            # registered local planner names, by policy_idx
int32[] agent_id
string[] name                # empty = unnamed
string[] handedness          # l | r, empty = right
string[] agent_type          # agent type name, empty = adult
string[] source_name         # releasing flow source, empty = spawned directly
int32[] source_seq           # release index within source_name
