import json, sys, csv
sys.stdout.reconfigure(encoding="utf-8")

with open("season_state.json", encoding="utf-8") as f:
    ss = json.load(f)
ss_names = set(ss["players"].keys())

with open("plofa_output/Pearls_vs_Claw_MD01/Pearls_vs_Claw_MD1.json", encoding="utf-8") as f:
    md1 = json.load(f)
md1_names = set(md1["players"].keys())

only_ss = ss_names - md1_names
only_md1 = md1_names - ss_names
both = ss_names & md1_names

print(f"Season state players: {len(ss_names)}")
print(f"MD1 (Pearls vs Claw) players: {len(md1_names)}")
print(f"Overlap: {len(both)}")
print()

if only_ss:
    print("=== In season_state but NOT in MD1 ===")
    for n in sorted(only_ss, key=lambda x: x.encode('ascii', 'replace').decode()):
        print(f'  "{n}"')

if only_md1:
    print("\n=== In MD1 but NOT in season_state ===")
    for n in sorted(only_md1, key=lambda x: x.encode('ascii', 'replace').decode()):
        print(f'  "{n}"')

# Check MD5 for player name overlap
with open("plofa_output/Pearls_vs_Triumpher_MD05/Pearls_vs_Triumpher_MD5.json", encoding="utf-8") as f:
    md5 = json.load(f)
md5_names = set(md5["players"].keys())

ss_not_md5 = ss_names - md5_names
md5_not_ss = md5_names - ss_names

print(f"\nMD5 players: {len(md5_names)}")
print(f"Overlap season_state vs MD5: {len(ss_names & md5_names)}")

if ss_not_md5:
    print("\n=== In season_state but NOT in MD5 ===")
    for n in sorted(ss_not_md5, key=lambda x: x.encode('ascii', 'replace').decode()):
        print(f'  "{n}"')

if md5_not_ss:
    print("\n=== In MD5 but NOT in season_state ===")
    for n in sorted(md5_not_ss, key=lambda x: x.encode('ascii', 'replace').decode()):
        p = ss["players"].get(n, {})
        print(f'  "{n}" (ss_matches={p.get("season_matches","N/A")})')

# Check season_matches for Pearls players who appear in MD5
print("\n=== Pearls players in MD5 - season_matches from season_state ===")
for n in sorted(md5_names):
    if n in ss["players"]:
        p = ss["players"][n]
        print(f'  {n}: matches={p["season_matches"]}, mins={p["season_minutes"]}')
    else:
        print(f'  {n}: NOT IN SEASON STATE')

# Check MD3, MD4 overlaps too
for md_label, md_path in [
    ("MD3 (Uditon vs Pearls)", "plofa_output/Uditon_vs_Pearls_MD03/Uditon_vs_Pearls_MD3.json"),
    ("MD4 (Oxton vs Pearls)", "plofa_output/Oxton_vs_Pearls_MD04/Oxton_vs_Pearls_MD4.json"),
]:
    try:
        with open(md_path, encoding="utf-8") as f:
            md = json.load(f)
        md_names = set(md["players"].keys())
        overlap = len(ss_names & md_names)
        md_not_ss = md_names - ss_names
        print(f"\n=== {md_label} ===")
        print(f"  Players: {len(md_names)}, Overlap with SS: {overlap}")
        if md_not_ss:
            print(f"  In match but NOT in season_state:")
            for n in sorted(md_not_ss, key=lambda x: x.encode('ascii', 'replace').decode()):
                print(f'    "{n}"')
    except FileNotFoundError:
        print(f"\n  {md_path} not found")
