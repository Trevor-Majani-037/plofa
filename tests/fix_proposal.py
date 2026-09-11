"""
Proposed fix for away-team shot map trajectory mirroring bug.

The issue: In plot_shot_map(), away-team shots have their trajectory
endpoints mirrored via eb_x = ((105 - ex) / 105) * 120, but ex (goal_x)
is 0.0 for away team, so eb_x becomes 120 (far right), creating a
horizontal line across the panel instead of pointing toward the goal.

The fix: For the away team, do NOT mirror the trajectory endpoint,
because the goal is already at the correct side (x=0 in original coords
maps to visual left after mirroring the start position).

Alternative: Change the mirroring to only apply to the start position,
not the trajectory end, so trajectories always point toward the goal.
"""

# Current code (lines 3013-3017 in exporter.py):
#     if home_panel:
#         eb_x = (ex / 105) * 120
#     else:
#         eb_x = ((105 - ex) / 105) * 120

# Proposed fix: Only mirror start position, keep end point direction
# toward the goal (which is at x=0 for away team in original coords)

print("Current away-team trajectory mirroring bug:")
print("  sx (original) = 93.9  -->  sb_x =", round(((105 - 93.9) / 105) * 120, 1))
print("  ex (goal_x) = 0.0     -->  eb_x =", round(((105 - 0.0) / 105) * 120, 1))
print("  Result: trajectory spans from sb_x~13 to eb_x=120 (horizontal line)")
print()
print("Proposed fix: Mirror only start, keep end pointing toward goal:")
print("  sx = 93.9  -->  sb_x =", round(((105 - 93.9) / 105) * 120, 1), "(mirrored start)")
print("  ex = 0.0   -->  eb_x = 0.0     (NO mirroring of end, goal stays at visual left)")
print("  Result: trajectory from x~13 toward x=0 (toward the goal)")
print()
print("OR: Mirror both but with reversed logic so goal stays on correct side:")
print("  sx = 93.9  -->  sb_x =", round(((105 - 93.9) / 105) * 120, 1))
print("  ex = 0.0   -->  eb_x =", round((0.0 / 105) * 120, 1), "= 0 (keep end at visual left)")
