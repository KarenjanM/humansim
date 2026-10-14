# One lifecycle edge of an interaction, stamped by its InteractionFrame
uint8 ACTIVATED=0     # reached min_participants, participants are bound
uint8 HOLD_ONSET=1    # every participant is on its slot, the hold starts
uint8 RELEASED=2      # hold ran its duration, participants are free
uint8 INTERRUPTED=3   # torn down before completing
uint8 CANCELED=4

int32 interaction_id
uint8 interaction_type       # arena_humansim InteractionType index
uint8 event
int32[] participants         # members at the time of the edge
