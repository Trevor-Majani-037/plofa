# Home team analysis
# For home (home_panel=True): sb_x = (sx / 105) * 120, eb_x = (ex / 105) * 120
# Home shots from Excel had x in various ranges, end_x varied

print('=== Home-team shot mirroring analysis (home panel, NO mirroring) ===')
print('For home: sb_x = (sx/105)*120, eb_x = (ex/105)*120')
print()

home_shots_sample = [
    (99.2, None),   # From Excel we saw various end_x values
    (103.0, None),
    (94.4, None),
    (87.3, None),
]

for sx, _ in home_shots_sample:
    sb_x = (sx / 105) * 120
    print(f'sx={sx:5.1f} -> sb_x={sb_x:6.1f}')

print()
print('Home team does NOT mirror x - shots keep original orientation')
print('Away team DOES mirror x - sb_x = ((105-sx)/105)*120')
print()
print('The mirroring is the problem for away team:')
print('  Away goal is at original x=0')
print('  After mirroring x -> (105-x), goal at x=0 goes to x=120')
print('  This means the away goal ends up on the WRONG side of the panel!')
