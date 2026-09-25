# Assumptions & Notes

## Sample Experiment
The current `configs/experiment_sample.yaml` implements the 3 steps provided:
1. Open the main box.
2. Take out the yellow box and place it on the left side.
3. Take out the red box and place it on the right side.

### Identified Ambiguities & Assumptions:
1. **Target Zones ("left side", "right side")**: "Left side" and "right side" are subjective, especially in microgravity where there is no fixed orientation. It's assumed the perception layer will define bounding box zones relative to a fixed reference point (e.g., the payload rack or the main box itself) to evaluate predicates like `object_position(yellow_box, left_side)`.
2. **Object States ("open")**: It's assumed the perception model is capable of classifying the state of the "main_box" as open or closed.
3. **Required Detectable Objects**: The perception layer will need to reliably detect and track the `main_box`, `yellow_box`, `red_box`, and the astronaut's hands.
4. **Step Transitions**: A step is marked as complete when all required predicates are met simultaneously or in sequence. We assume placing the box means the box stays there, and no further constraints like "hands are removed from the box" are strictly required unless specified later.
