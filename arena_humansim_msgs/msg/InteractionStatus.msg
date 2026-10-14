# Snapshot of one live interaction
uint8 FORMING=0
uint8 ACTIVE=1

int32 interaction_id
uint8 interaction_type       # arena_humansim InteractionType index
uint8 outcome                # FORMING | ACTIVE
int32[] participants
int32[] queue
bool arrived                 # every participant is on its formation slot
bool holding                 # HOLD_ONSET fired, interaction not yet ended
float64 hold_elapsed         # seconds of hold counted against duration
float64 duration             # hold duration in seconds, -1 = unbounded
