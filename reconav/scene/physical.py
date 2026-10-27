"""Single physical contract shared by manual-GT construction and navigation."""

ROBOT_DIAMETER_M = 0.50
ROBOT_RADIUS_M = ROBOT_DIAMETER_M / 2.0
ROBOT_HEIGHT_M = 1.50
# Uniform undercarriage clearance.  The complete scene mesh remains the
# collision source; geometry below this height is physically below the body,
# not semantically whitelisted.
BODY_BOTTOM_OFFSET_M = 0.10
ENDPOINT_OBJECT_CLEARANCE_M = 1.00
# Numerical tolerance for semantic-distance comparisons only. This is many
# orders of magnitude below scene resolution and does not practically enlarge
# the 1 m contract; it only prevents solver/replay round-trip disagreement.
SEMANTIC_DISTANCE_EPS_M = 1e-8


def within_semantic_clearance(distance_m: float, clearance_m: float) -> bool:
    return float(distance_m) <= float(clearance_m) + SEMANTIC_DISTANCE_EPS_M


# A route either encounters an object or it does not.  Hard COUNT/AVOID,
# PREFER and evaluator replay must not assign different spatial truth values.
OBJECT_ENCOUNTER_CLEARANCE_M = 1.00
# Kept as a source-compatible alias for callers outside the current benchmark.
PREFERRED_OBJECT_CLEARANCE_M = OBJECT_ENCOUNTER_CLEARANCE_M
MIN_PUBLISHED_PORTAL_CLEAR_WIDTH_M = ROBOT_DIAMETER_M

# Existing non-manual data sources import these names.  They are aliases to the
# one physical contract, not separately configurable defaults.
DEFAULT_ROBOT_DIAMETER_M = ROBOT_DIAMETER_M
DEFAULT_ROBOT_RADIUS_M = ROBOT_RADIUS_M
