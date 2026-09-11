import json

# Load the Excel shot map data we already extracted
# and verify the mirroring logic

# From the Excel we saw: away team shots have x in 82.9-101.6, End X = 0.0
# And Excel end_x was 0.0 for away shots

# The plot code does this for away (home_panel=False):
#   sb_x = ((105 - sx) / 105) * 120
#   eb_x = ((105 - ex) / 105) * 120

# Test with actual away-shot data from Excel:
away_shots_data = [
    (82.9, 0.0),
    (93.9, 0.0),
    (101.1, 0.0),
    (89.3, 0.0),
    (91.0, 0.0),
    (88.4, 0.0),
]

print('=== Away-team shot mirroring analysis ===')
print('orig sx     end_x     sb_x    eb_x    dx')
print('-'*50)

for sx, ex in away_shots_data:
    sb_x = ((105 - sx) / 105) * 120
    eb_x = ((105 - ex) / 105) * 120
    dx = eb_x - sb_x
    print(f'{sx:8.1f} {ex:8.1f} {sb_x:8.1f} {eb_x:8.1f} {dx:8.1f}')

print()
print('Problem: eb_x =', eb_x, '(far right, should be near 0 toward away goal)')
print('sb_x =', sb_x, '(left side) creates line spanning entire panel')
print()
print('Expected: both sb_x and eb_x near 0, pointing toward away goal')
