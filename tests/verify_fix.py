# Verify the fix: away-team trajectory should now point toward goal

print('=== Verification of away-team trajectory fix ===')
print()

# After fix: eb_x = (ex / 105) * 120 (same as home, no mirroring)
# With ex = 0.0 for away team shots:

for sx in [82.9, 93.9, 101.1, 89.3, 91.0, 88.4]:
    ex = 0.0  # away team: goal_x = 0.0
    sb_x = ((105 - sx) / 105) * 120  # mirrored start (unchanged)
    eb_x = (ex / 105) * 120  # FIXED: no mirroring of endpoint
    dx = eb_x - sb_x
    print(f'sx={sx:5.1f} ex={ex:5.1f} sb_x={sb_x:7.1f} eb_x={eb_x:7.1f} dx={dx:8.1f}')

print()
print('Result: dx is now NEGATIVE (', end='')
for sx in [82.9, 93.9, 101.1, 89.3, 91.0, 88.4]:
    ex = 0.0
    sb_x = ((105 - sx) / 105) * 120
    eb_x = (ex / 105) * 120
    dx = eb_x - sb_x
    print(f'{dx:.0f}', end=' ' if sx != 88.4 else '')
print(') toward the goal (visual left), not horizontal across the panel')
print()
print('Trajectories now point from shot position toward the away goal,')
print('which is at visual x≈0 on the away panel - correct!')
