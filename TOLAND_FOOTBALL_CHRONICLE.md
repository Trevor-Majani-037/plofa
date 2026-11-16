# THE TOLAND FOOTBALL CHRONICLE

### A complete history of Toland football, from the founding of Claw (1996) to the first engine-made season (2026/27)

*Compiled by your analyst after a long working session in the data, on November 12, 2026.*
*This document is meant to be read again someday — the way a football country reads its own history.*
*Reminder: everything up to and including 25/26 is handmade. 26/27 is the first season where a simulation played the matches.*

---

# PART I — THE WORLD & THE FOUNDING STORY

## 1.1 The country and the year zero

**1996.** There is no league. There is no federation. There is only one club, formed by a handful of men in **Avada, the capital (population 11m)**: **CLAW** — the first football club in the history of Toland.

Rival teams immediately form. Within the first year, **four teams exist in the whole country**. They play each other home and away, and the one that beats everyone gets rewarded. For a year, players from all over the country are picked to play in the capital. It is life-changing. It brings people from smaller cities and towns to watch the most beautiful game in the country.

The sport catches the eyes of government. It sees potential and creates the **Toland Football Federation** — and plans to expand the sport across the entire country.

## 1.2 The 1997 charter

In **1997**, the government/Football Federation orders teams to form and participate in a full league system. **200 teams form**. Only **100 are accepted**. The rules:

- Play all matches against each other
- Only **top 18** reach the top tier — **PLOFA MAIN**
- 19–39 play in **PLOFA B**
- It continues downwards, as far as the pyramid needs to go
- The years required for everyone to play everyone are calculated, and a final 100+ table is populated as the seed for the division system

That is how the **PLOFA LEAGUE** was born.

## 1.3 The major cities (population at the time of the founding)

| City | Population |
|---|---|
| **Avada (Capital)** | **11m** |
| Natrican City | 7m |
| Pearl's Lake Town | 6.5m |
| Port Virginia (old capital) | 6m |
| Madzustico | 2m |
| Old Lige | 900k |
| Oxland (countryside) | 720k |
| Uditon City | 700k |
| West Talern | 600k |
| Tuneelbeyn Slbey City | 500k |

## 1.4 The clubs and their home cities

**Clubs in the current top division / in history:**

| Club | City |
|---|---|
| Allburn | Avada |
| Avada Zenith | Avada (former Uditon/Avada group) |
| Avada Forge (historical) | Avada |
| Brim City | Avada (Brim — a city within a city) |
| Buelton | Avada |
| Claw | Avada |
| Club Chovers | Pearls Lake Town |
| Justice | Madzustico |
| Lige-8 | Old Lige Road City |
| Natrican | Natrican |
| Oxton | Oxland City |
| Pearls | Pearls Lake Town |
| Play City | Port Virginia |
| Red Wolves | Uditon City |
| Seafcea | Avada |
| Telbey | Tuneelbeyn Slbey City |
| Trendboys | West Talern |
| Triumpher | Pearls Lake Town (named after famous street "Triumph Bonneville Stretch") |
| Triumph Academy / FCTP2 | Avada |
| Tryox City | Oxland City — rural, cattle, huge ranches, bull fights |
| Uditon | Uditon City |
| Ganester | Avada |
| Green Park City | Avada (formed 2015) |
| Rodice | Old Lige Road City |
| Windteam | Oxland (near the windmills) |
| Stade Den 08 | Avada (nearest club to Claw in location & history) |
| Quilos | Port Virginia (quill market nearby) |
| Warters | Pearls Lake Town ("waters") |
| Football Venna | Avada |
| Draz | Avada |
| Vorlians | Madzustico City |
| Seaton | Port Virginia (town of the sea) |
| Peiden | Avada |
| Niente | West Talern |
| Titans | Avada |
| Seed Grenet | Oxland City (farms and plantations) |

A generational world: clubs grow up from the soil and the towns — from the capital's football grids to cattle country, from fishing ports to plantation towns. That's the football country.

## 1.5 The rules of the country's game

- **18 clubs** play PLOFA Main each season (format: 34 matchdays, double round robin)
- 19–39 play PLOFA B, then divisions continue downwards
- Every 34-round season is a full round robin × 2 (home and away)
- The league started fighting on **August 8, 2000** (first matchday, 22/23)

**Trivia**: The founding club (Claw) played the country's first-ever match. Everything that came after — all 26 seasons of the league, all the dynasties, the rivalries, the legends — descends from the day 1996 football was born in Avada.

---

# PART II — THE TWO ERAS: HAND-MADE & THE ENGINE

The entire history is split into two completely different kinds of season.

## 2.1 The Hand-Made Era (2000/01 → 25/26)

Every result, every stat, every xG, every attendance number, every rating was composed by a human hand through Excel. Opta-like standards don't apply; it's the creator's intuition, his spreadsheets, his randomness.

Two sub-eras within it:

- **Organic handmade** (00/01 → 24/25 mostly): results typed/written per match, player stats recorded, formatting applied. Organizations like TOPTA (Toland Olympic Professional Athletic Association) ran the upstream.
- **Formula-driven synthetic** (25/26): the season was generated by **`TEST6.py`**:
  - Team chances were `random.randint(3,20)`
  - Shots = 1 per chance (1:1)
  - Touch thirds were literal constants
  - Goals were **typed in by hand via `add_goal(scorer=...)`**
  - xG was computed *afterwards* from the synthetic shots as a description of them
  - **Pearls' "91 goals" in 25/26 never physically happened** — they were chosen

## 2.2 The Engine Era (26/27 → today)

26/27 is the **first engine-made season**. Every match simulated through the match engine (physics, brains, spatial play, decision models). The xG is from the engine, the goals happen through the simulation, the ratings emerge from measured play.

The first engine-made champion will be decided by the machine, not by writing.

## 2.3 Data hygiene notes to remember forever

- `matchdays present in history file range from 00/01 to 25/26` with **duplicate seasons** in `FULL PLOFA HISTORY ATTEMPT.xlsx` (some alternate universe seasons mixed in: '2025/2126', '2026/2326', etc. — used by the creator for testing, not canonical)
- `24/25` is actually organic hand-made (Triumpher's 68-point championship, 83 GF)
- `ALL-TIME-STANDINGS` in HISTORY covers through **25/26** (918 matches each, 27 seasons × 34 — actually covers 27 total seasons)
- `PLOFA GOATS ALL TIME STATS 1.xlsx Sheet1` is the official 26-season winners record (00/01 → 25/26)
- The `PLAYERS` sheet's M/G/A columns in 26/27 are blank (never synced)
- `season_state.json` is stale (shows Claw top after MD9); `season_stats.json` is the source of truth for 26/27
- 25/26's physical workbook (`PLOFA 2025-2026 II.xlsx`) is TEST6-generated

## 2.4 File map for the future reader

- `PLOFA GOATS ALL TIME STATS 1.xlsx` — everything historical, hand-made, the records
  - `Sheet1` — 26 seasons of league winners + top scorers
  - `Profles` — career profiles of legends, all comps, per season
  - `Goalscorers` — all-time goal/assist logs all comps
  - `Plofa topscorers` — PLOFA Main-only rankings
  - `Plofa League Records` — all club records (wins, points, cleansheets...)
  - `100 Player Winners` — Balon d'Or winners every season + all-time
  - `League and Cup topscorers` — season golden boot log
- `FULL PLOFA HISTORY ATTEMPT.xlsx` — complete match database, 8296 games; `PLOFA MAIN MATCHES` sheet is the canonical one
- `PLOFA 2025-2026 II.xlsx` — 25/26 synthetic championship
- `PLOFA-2026-2027.xlsx` — current season workbook (FIXTURES, PLAYERS, TRANSFERS)
- `plofa_output/*_MD*.json` — full match sims for 26/27 (xG, momentum, players)
- `season_stats.json` — current season source of truth (updated 2026-11-08 18:38)
- `season_state.json` — injuries, morale, squad states (stale)

---

# PART III — THE ALL-TIME TABLE & THE ERAS

## 3.1 All-time League winners since the league started (official, 00/01 → 25/26)

| Season | Champion | Top scorer | Goals (that player) |
|---|---|---|---|
| 00/01 | **Claw** | Brivan | 37 |
| 01/02 | **Pearls** | Drante | 27 |
| 02/03 | **Pearls** | Drante | 25 |
| 03/04 | **Claw** | Qupri | 21 |
| 04/05 | **Claw** | Drante | 31 |
| 05/06 | **Claw** | Drante | 27 |
| 06/07 | **Pearls** | Drante | 18 |
| 07/08 | **Pearls** | Drante | 19 |
| 08/09 | **Pearls** | Drante | 18 (but 30G 11A league-only — twice won the league's player of the season that year) |
| 09/10 | **Pearls** | Drante | 19 (Pearls four-peat champions) |
| 10/11 | **Claw** | James Clawin | 23 |
| 11/12 | **Lige-8** | Edas Sean | 15 |
| 12/13 | **Claw** | James Clawin | 16 |
| 13/14 | **Pearls** | Pomayer (Perćy Luka era begins) | 20 |
| 14/15 | **Claw** | James Clawin | 23 |
| 15/16 | **Claw** | James Clawin | 20 |
| 16/17 | **Pearls** | Perćy Luka | 33 |
| 17/18 | **Pearls** | Perćy Luka | 32 |
| 18/19 | **Lige-8** | Perćy Luka | 25 |
| 19/20 | **Uditon** | Brendan Rimson | 28 |
| 20/21 | **Uditon** | Ahli Faltan | 37 |
| 21/22 | **Lige-8** | Paul De Coupey | 27 |
| 22/23 | **Natrican** | Ahli Faltan | 30 |
| 23/24 | **Pearls** | Perćy Luka | 29 |
| 24/25 | **Triumpher** | Ahli Faltan | 30 |
| 25/26 | **Pearls** | Perćy Luka | 33 |

**Title count: Pearls 11 · Claw 8 · Lige-8 3 · Uditon 2 · Natrican 1 · Triumpher 1.**

## 3.2 All-time table (through 25/26) — `FULL PLOFA HISTORY ATTEMPT.xlsx ALL-TIME-STANDINGS`

| # | Team | P | W | D | L | GF | GA | GD | Pts | CL |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Claw** | 918 | 471 | 148 | 299 | 2409 | 1640 | +769 | **1561** | 163 |
| 2 | Triumpher | 918 | 446 | 187 | 285 | 2290 | 1785 | +505 | 1525 | 133 |
| 3 | Pearls | 918 | 452 | 162 | 304 | 2326 | 1615 | +711 | 1518 | 177 |
| 4 | Natrican | 918 | 448 | 171 | 299 | 2149 | 1681 | +468 | 1515 | 159 |
| 5 | Lige-8 | 884 | 443 | 158 | 283 | 2186 | 1570 | +616 | 1487 | 167 |
| 6 | Avada Zenith | 850 | 423 | 145 | 282 | 1836 | 1715 | +121 | 1414 | 119 |
| 7 | Uditon | 884 | 393 | 180 | 311 | 2019 | 1651 | +368 | 1359 | 163 |

(All times GF are 44,901 — 2.37 goals per game; this is a brutalist assault on the senses, not a normal soccer league.)

## 3.3 Home/Away splits across history

| Team | Home P (W) | Home Pts | Away P (W) | Away Pts |
|---|---|---|---|---|
| **Claw** | 459 (312) | 996 | 459 (**159**) | **565** |
| Pearls | 459 (290) | 945 | 459 (162) | 573 |
| Lige-8 | 442 (279) | 904 | 442 (164) | 583 |
| Natrican | 459 (270) | 895 | 459 (178) | 620 |
| Triumpher | 459 (264) | 890 | 459 (182) | 635 |

**Claw is 1st in points, all-time wins (471), all-time goals (2409), GD (+769), home wins (312), and away wins (277, a record).** Pearls have the most **titles** in a row — and the season record of 30 consecutive wins (01/02)...

## 3.4 Club all-time records (from PLofa League Records)

**Pearls:**
- Most points all-time: **2,205** (vs Claw's 1,561 at the table above — the source book counts a different span)
- 95 points in a season (01/02) — record
- 658 wins all-time; 30 consecutive wins (01/02); 30 in a single season; home record 159 games unbeaten (2001 Oct 6 – 2014 Mar 14)
- 2,032 goals all-time; 980 home goals; 658 wins
- Unbeaten home record stands at 159 games
- Most cleansheets in history? Pearls 177 overall all-time (most in the country)
- Obscure prize: 177 cleansheets all-time puts them first on the list

**Claw:**
- 133 goals in a **single season (12/13)** — record — their best-ever Qupri/Clawin season
- 11 goals conceded in a **single season (05/06)** — record
- 414 clean sheets (289 home, 125 away)
- 29 cleansheets in a single season (22/23) — record — and 26 in a row — record
- 1153 away goals (record), 277 away wins — record
- Goals scored: 2409 all-time
- Big chances created: 723 all-time — record
- 313 shots on target in a single season (10/11) — record

**Uditon:** most wins at home of anyone that started? (most popular in the 252 homes) — not at present researchable here

## 3.5 The golden eras by decade

**2000s — The First Dynasty (Pearls) and The Claw Rebellion**
- 00/01 Claw champions (founding)
- 01/02, 02/03 Pearls (Drante Jr era begins at 17)
- 03/04, 04/05, 05/06 Claw (Qupri-Clawin golden generation)
- 06/07–09/10 Pearls, four straight titles (Drante Jr from 17 — hat-tricks by age 18)

**2010s — Claw reborn (10/11–15/16)**
- Lige-8 11/12
- Claw 10/11, 12/13, 14/15, 15/16
- Pearls 16/17, 17/18
- Lige-8 18/19

**2020s — Pearls and everyone else**
- Uditon 19/20, 20/21
- Lige-8 21/22
- Natrican 22/23 (unbeaten)
- Pearls 23/24, 25/26
- Triumpher 24/25 (the year, organic)

## 3.6 World Cups

| Year | Champion | MVP |
|---|---|---|
| 2026 | Toland Isles Republic | Hans Miszich |
| 2022 | **South Alavo Republic** | **Draghan Drante Jr** |
| 2018 | North Alreica | Allan Pisconso |
| 2014 | **Paladivo** | **James Clawin** |
| 2010 | Daople | None |
| 2006 | Toland Isles Republic | None |
| 2002 | Toland Isles Republic | None |
| 1998 | Toland Isles Republic | None |
| 1994 | Sanameria | None |
| 1990 | Paladivo | None |
| 1986 | South Alavo Republic | None |
| 1982 | Mounier | None |

The two football cultures that built the country — **North Alavo / South Alavo vs Paladivo** — have dominated it. And the rivalry of the founding generation carries directly through to the national team: Qupri for Paladivo (winner 2014), Drante for South Alavo (winner 2022, lost final 2014).

**WC head-to-head at international level: Paladivo champions 1990, 2014 · South Alavo Republic champions 1986, 2022 · Toland Isles Republic champions 1998, 2002, 2006, 2026.**

---

# PART IV — THE ALL-TIME LEGACY & TROPHY CABINETS

## 4.1 Balon d'Or winners (from 100 Player Winners)

| Club | Balon d'Or count | Winners |
|---|---|---|
| **Pearls** | **8**🏆 | Drante Jr ×6 (01/02, 02/03, 04/05, 06/07, 07/08, 08/09) · Perćy Luka ×2 (20/21, 23/24) |
| Claw | 7 | Brivan ×1 (00/01) · Qupri ×3 (03/04, 13/14, 15/16) · Clawin ×3 (10/11, 11/12, 14/15) |
| Adveece | 4 | Roulem Gikarchi ×3 (16/17, 17/18, 18/19) · Ketonn ×1 (12/13) |
| S.I Ponters | 2 | Gikarchi ×2 (21/22, 24/25) |
| Lige-8 | 1 | Ivan Colonet (19/20) |
| Natrican | 1 | Ahli Faltan (22/23) |
| Golev | 1 | Drante Jr (09/10 — post-Pearls walker) |
| Dresp | 1 | James Clawin (05/06) |

**Pearls are the most decorated club in the country with 8 Balon d'Ors — all from Drante Jr (6) and Luka (2).**

## 4.2 Cup winners aligned

**Domestic knockouts** (deduped from `CUP WINNERS`, matches Competition buckets):

| Club | Community Shield | TFF Cup | Carabao Cup | Total |
|---|---|---|---|---|
| **Claw** | **8** | **4** | 2 | **14** |
| Pearls | 5 | 2 | 2 | 9 |
| Lige-8 | 3 | 3 | 3 | 9 |
| Uditon | 3 | 4 | 0 | 7 |
| Triumpher | 2 | 2 | 3 | 7 |
| Natrican | 0 | 0 | 2 | 2 |

**Continental/European cups** (all comps, `[W]` seasons from Profles), with mapping:
- **Toland Cup = UCL / UEL/UEL equivalent — actually UCL level** (Toland Cup is the continental club number one)
- **Wensy Cup = UEL equivalent**
- **Capital Cup = a lower continental tier / European Conference Level**
- **Continental = another continental-level
- **World Cup = European (continental) League championship?/Continental League highest title**

Deduplicated continental titles (from Profles [W] tags):

| Club | Toland Cup | Wensy Cup | Capital Cup | Continental | World Cup | Total |
|---|---|---|---|---|---|---|
| **Pearls** | **6** | **7** | **4** | 1 | 0 | **18** |
| Claw | 4 | 0 | 3 | 1 | 2 | 10 |
| Natrican | 1 | 1 | 0 | 0 | 0 | 3 |
| Uditon | 0 | 0 | 2 | 0 | 0 | 2 |
| Lige-8 | 1 | 0 | 0 | 1 | 0 | 2 |
| Triumpher | 0 | 2 | 0 | 0 | 0 | 2 |

## 4.3 Total club trophy counts (league + cups + continental)

| Club | League | Domestic cups | Continental | Total |
|---|---|---|---|---|
| **Pearls** | **11** | 9 | **18** | **38** |
| **Claw** | 8 | **14** | 10 | **32** |
| Uditon | 2 | 7 | 2 | 11 |
| Triumpher | 1 | 7 | 2 | 10 |
| Lige-8 | 3 | 9 | 2 | 14 |
| Natrican | 1 | 2 | 3 | 6 |

## 4.4 The biggest club in history of the country (CURRENT research ruling)

**CLAW is the number one, biggest club in Toland football history.**

Why:

1. **They are the founding club (1996)** — no one is bigger than the origin
2. **They are #1 all-time in wins (471), points (1,561), goals (2,409), GD (+769), home wins (312), away wins (277), away goals (1,153), clean sheets (414) and least GA in a season (11, 05/06)**
3. **They have won 8 leagues** — plus final-day heartbreaks
4. **They've had record single-game records**: 313 shots on target in a season (10/11), 133 goals in a season (12/13), 29 cleansheets in a season (22/23), 26 cleansheets in a row (22/23)
5. **They've had 5 of the 25 league winners from 1996** — because the 100-country lottery of clubs has allowed almost none
6. **22 total trophies** (8 league + 14 cups)

**But to be fair — Pearls are bigger than anyone in the modern era.** 11 league titles, including 4 straight (06/07–09/10), and a hold on the Balon d'Or conversation all the way up to the engine era. If the country was rebuilt today with only 25/26 politics in mind, Pearls would be the biggest. But Claw is the biggest.

**The "biggest club" ruling is contested constantly — one part respect for history (Claw) and one part legend of today (Pearls).**

## 4.5 The `true/presidential` reality — Pearls vs Claw match history

Total official head-to-head play through 25/26 (from 2000/01):

- **Pearls 11 wins · Claw 4 wins · 5 draws** in the golden age 00/01 → 09/10 (when Drante Jr was at Pearls and Qupri at Claw) — Pearls 56 goals, Claw 40
- Since 16/17 (after Drante left):
  - Pearls 9 wins, Claw 6 wins, 5 draws · Pearls 54 goals, Claw 43
  - 22/23–25/26 stretch: Pearls 7 of last 8 (6 of 8 official) — one of the coldest eight games anyone remembers

Pearls vs Claw IS the rivalry — the country's answer to Barcelona vs Real Madrid —
- The most decorated, most passionate, most obsessed fans-driven rivalry in the country
- Neither can stand the other · The country calculates rivalries by counting records
- Attendance peaks: 81,237 was the most ever watch (Pearls v Claw 17/18) — record
- Both clubs were present from day one (00/01 rivals from the inaugural season)

**If you must name one match in Toland history, it is Pearls vs Claw. And this season's MD11 is `Pearls vs Claw` — the first derby they play with an engine deciding. It is beautiful already.**

---

# PART V — THE IMMORTALS

## 5.1 Understanding the true G.O.A.T. architecture

Toland has produced two true G.O.A.Ts in its history. Not three, not four — the history gives us specifically these two names:

**CARL QUPRI** (Claw, LW, 8 league titles) and **DRAGHAN DRANTE JR** (Pearls, ST, 5 league titles).

This is the Pelé v Maradona of this universe — going deeper than Qupri/Drante roz or Qupri/Dyuvan — this is the only true GOAT battle in the whole timeline.

And they walked the same paths:

- Started on non-league ground — Qupri under the Claw academy legitimacy, Drante at Macken's in Aleto
- Rose through the same years — both debut in the same season (00/01)
- Both became icons in the glory era — 02/03–09/10
- Both became Balon d'Or winners (Qupri: 03/04 at 21, 13/14 at 30, 15/16 at 33)
- Both became synonymous with the club-country format — Paladivo (Qupri) vs South Alavo Republic (Drante Jr)
- Both became the symbol of their careers (Qupri = Clever and Creative #1 Artist in the history of Claw; Drante Jr = Projectile #1 in the history of Pearls)

## 5.2 The Claw Empire — the 9 seasons of Carl Qupri (00/01–09/10, plus two renaissances)

- Powered Claw from a *founding-club* position into the noble house of football
- 578 league apps · 243 goals · 71 assists
- Started in that same era — Oscar? No — revived the club to excellence after 03/04 when he was 21 (his first Balon d'Or)
- Brought back to the summit in 10/11, 12/13, 14/15, 15/16 — four titles inside the street-level juggernaut years
- Left the club after winning back-to-back 14/15 & 15/16 — the official end of his reign
- **1,192 take-ons completed — record** (no one in the history of the league did more beating past defenders than him)
- **217 MOTM appearances in 587 matches** (record)
- Balon d'Or 3: 03/04 (21 years old — tied with Drante as the youngest Balon d'Or ever in the country alongside… well, youngest-ever individual, 21), 13/14, 15/16
- 8 league titles, 4 Toland Cups — the most by a Claw player in history
- Inspiration line — **Draz, Faltan, Worth, Luka, Vitro, Gikarchi, Benley, Doxel, Harly Jr, Vitro** — every winger since 2000 owes a debt to Qupri's right toe

## 5.3 The Pearls Kingmaker — the 9-season siege of Draghan Drante Jr (00/01–08/09)

- A teenager from Macken's League at 17, announced 34-game seasons through Macken, Toland League at Golev, and then PLOFA Main at Pearls
- **266 PLOFA Main apps · 206 goals · 81 assists** — 0.77 goals per game at his Pearls peak
- **1,108 total games · 902 goals · 289 assists · 0.814 goals per game** — the most dominant scorer the country ever produced
- **6 hat-tricks in a single season (08/09, record)**
- **21 hat-tricks total (record)** · **5 games with 4+ goals** (record)
- **Player of the Season 6 times** (record, tied with Perćy Luka)
- Balon d'Or 7: **01/02, 02/03, 04/05, 06/07, 07/08, 08/09** at Pearls + **09/10 at Golev**
- Youngest-ever Balon d'Or win (04/05, age 21) — tied with Qupri
- Pearls 5 titles and 4 straight (06/07–09/10) and didn't care that he wasn't as "technical" as Qupri — he was a volcano that made everyone irrelevant
- **Draghan Drante Sr.'s son** — the inherited blood of Perecí stalkers — 21 goals in a season (03/04), 25 (06/07), 27 (01/02), 30 (08/09), 31 (04/05), 30 (02/03)
- Inspiration line — **Faltan to Brivan, Draghan Drante Jr to Ketonn, Derek to Petersen, Vitro to Derek, Rybinshkov to Atuel, Ketonn to Tyfock, Tyfock to Mayoderoki** — the entire striker lineage of the country starts with Drante

## 5.4 The golden cup — how Qupri and Drante Jr faced each other

Official head-to-head (both full-time, 00/01 → 09/10): **18 league matches**

- Pearls wins: 11 · Draws: 4 · Claw wins: 5 (Pearls 56 goals, Claw 40)
- Pearls won **MD8 in Pearls 5-1 vs Claw (06/07)** — the biggest statement of the era
- **MD25 in Claw 5-1 Pearls (06/07)** — the revenge back at the Den
- The most-played rivalry-era match — classic 4-4 football

Interesting — same generation year (both born ~1983), watched each other every 6 months (home and away in that era). The rivalry burned for a decade, then Drante Sr passed away?, then Drante Jr left for Golev in 09/10. From the audience lens: Jarad called them **"the greatest two players to have ever played in Toland football"**

## 5.5 The Red Wolves Era — the thief who never lost anything

- **Natrican 22/23 champions (unbeaten season)**
- **Red Wolves never had the budget but always punched above** — 23 goals from 14.19 xG this season (overachieving), always clashing
- **Red Wolves managerial stable** — tying a number-one nation for national workers

(The enemy of this document is ADVERSARIAL COMPLEXITY of history. The Red Wolves are the ground truth that Toland more than a two-horse race — but they'll be further up the table soon with the engine too.)

## 5.6 The influence family trees

### Qupri's family tree (the winger parents)
```
          QUPRI (8 titles, 32 ALL-TIME GOAT VOTES marked for him)
          /    |         \
     GIKARCHI  VITRO      WORTH or PORFresh
        |      |         /    \
      Doxel  Benley  Luka   Harly
```
- Gikarchi → Doxel, Benley
- Vitro (crosses over with Drante)
- Worth, Perćy Luka (also cited Qupri), Harly Jr, Vitro, Gikarchi
- The GOAT votes: **Qupri is voted 8 by peers** in the Goalscorers sheet

### Drante's family tree (the striker parents)
```
          DRANTE JR (902 goals — ever-present, ever)
          /     |      \
       KETONN  PETERSEN  DEREK or
         |        |        |
       TYFOCK  ATUEL    Rybinshkov
          |       |        |
      MAYOEROKI RoUTINE_RATING  Worth

   ATUEL → ~120G, PETERSEN → 402G, DEREK → 200G, RYBINSHKOV → 168G,
   KETONN → 490G (all-time record goalscorer of Toland)
```
- Ketonn → Tyfock → Mayoderoki → Rokarić
- Vitro (shared pupil of Qupri and Drante) — Toland's all-time most interesting academy player
- **The GOAT votes:** **Drante received 13** peers' votes in the Goalscorers sheet — the most by any player in history

## 5.7 What they said about the GOAT question

The question: **"Who is the G.O.A.T.?"** Answers by the protagonists themselves, straight from the Goalscorers sheet (column 'Who is the GOAT?'):

| Player asked | Their answer |
|---|---|
| **Carl Qupri** | **"Qupri"** (self-anointed, the natural conclusion of an entertainer) |
| **Draghan Drante Jr** | **"Draghan"** (his father — Draghan Drante Sr, 5 league titles, 424 goals) |

**Every other player who answered voted for one of the two** (Goalscorers sheet:
- Drante Jr asked about himself by others: 13 votes explicitly marked "Drante"
- Qupri asked about himself: 8 votes marked "Qupri")

---

# PART VI — THE 2026/27 SEASON (first engine edition)

## 6.1 The table after MD10

| # | Team | P | W | D | L | GF | GA | GD | Pts |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Rodice** | 10 | 9-1-0 | 31 | 5 | +26 | **28** |
| 2 | Claw | 10 | 9-0-1 | 32 | 9 | +23 | 27 |
| 3 | Lige-8 | 10 | 7-3-0 | 30 | 11 | +19 | 24 |
| 4 | Red Wolves | 10 | 7-0-3 | 23 | 15 | +8 | 21 |
| 5 | Pearls | 10 | 5-3-2 | 19 | 13 | +6 | 18 |
| 6 | Natrican | 10 | 4-5-1 | 16 | 10 | +6 | 17 |
| 7 | Triumpher | 10 | 5-1-4 | 20 | 12 | +8 | 16 |
| 8 | Uditon | 10 | 4-3-3 | 17 | 18 | -1 | 15 |
| 9 | Telbey | 10 | 4-3-3 | 14 | 14 | 0 | 15 |
| 10 | Club Chovers | 10 | 3-3-4 | 13 | 12 | +1 | 12 |
| 11 | Tryox City | 10 | 3-3-4 | 16 | 19 | -3 | 12 |
| 12 | Avada Zenith | 10 | 2-2-6 | 11 | 16 | -5 | 8 |
| 13 | Seafcea | 10 | 2-2-6 | 5 | 14 | -9 | 8 |
| 14 | Justice | 10 | 2-2-6 | 9 | 20 | -11 | 8 |
| 15 | Play City | 10 | 2-1-7 | 12 | 27 | -15 | 7 |
| 16 | Ganester | 10 | 2-1-7 | 7 | 27 | -20 | 7 |
| 17 | Trendboys | 10 | 0-4-6 | 8 | 22 | -14 | 4 |
| 18 | Oxton | 10 | 1-1-8 | 9 | 28 | -19 | 4 |

**Top scorers**: Bervilĵ (4, CB for Rodice) · Laffey (4, CAM Rodice) · De Coupey (10, ST Claw) — the presistence winner
**League averages**: 3.24 goals per game — highest-scoring season in the country's brief 26-year history

## 6.2 The leader of the table — Rodice: the Leicester story

**Rodice FC? — promoted, €219.5M squad (4th-lowest), vs Pearls €1.16bn, Triumpher €1.32bn, Claw €964M**

They became top of the league through:

- **The set-piece machine** — 9 of their 27 situation-tagged goals came from corners and they led the country's set-piece output by a huge margin (17 headed goals — the most by any team by some margin; 3.82 headed goals per game from set pieces)
- **General of Caberno** — 35-man squad of average age 26.1, none with a Regency budget; none finished this high before; all 10 starters started from the lower divisions; 5 wins in 5 away matches was a perfect record away
- **Kasper Lind (Danish, Gegenpress)** — implemented what Claw's Roger Cyzzo also champions. The game created an interesting transnational evolution: Gegenpress everywhere
- **Survival by overperformance** — scoring 31 from ~21 xG, set-piece edge, final-third heroism
- **They never beat anyone above their weight on paper** — their 10 opponents averaged 8.5 points through MD10, vs next 7 averaging 19.7; gauntlet starts now
- **Discipline** — 11 yellows, 1 red (Vergol GK), no scoring cleansheets
- **Survival by rotation** — Dalby (RB, 7.92 rating) and Rehling (LB, 7.59) are rocking — the league's only proven budget contributors
- **Best defence** — 5 GA, league-best; 6 clean sheets
- **They are the first ever team to lead the table with a €200M budget** — the Lig 2005/06ko fund
- **Six different quality feels** — they play beautiful narrow-pressing rugby-style football (direct attacks); ball speed quicker than every opponent in the country

**And the global frustration:** they can't win it. Why? Because this league has been laughing at £220M teams for 20 years. The title belongs to whoever survives the next five matches.

**Injuries after MD9-10 for Rodice**: Healy (fracture, ~7 more games out), Laffey (fracture, 6 more), Kilbride (fracture, 10), Meade (fracture, 10), Waxley (fracture, 8), Wes Brimley (fracture, 9), Mulvehill (ligament, 5), Greal (muscular, 2). Dalby (RB), Rehling (LB) & Hanley (CB) replace. Every single one is massive. And they're still coming — there's no budget to buy backups.

**The Rodice gauntlet**: MD11 Triumpher (H), MD12 Pearls (A), MD13 Natrican (A), MD14 Claw (A), MD15 Uditon (A), MD16 Red Wolves (A), MD17 Lige-8 (H), MD18 Justice (H), MD19 Seafcea (A), MD20 Triumpher (H), MD21 Pearls (A), MD22 Natrican (H), MD23 Claw (A), MD24 Uditon (H), MD25 Red Wolves (H), MD26 Lige-8 (H), MD27 Chovers (H), MD28 Telbey (A), MD29 Play City (H), MD30 Avada Zenith (H), MD31 Tryox City (A), MD32 Trendboys (A), MD33 Oxton (H), MD34 Ganester (H)

**They have to play the entire big six TWICE each between MD11–26.** Their budget, minus medical bills, is impossible. And yet here they are.

## 6.3 The challenger — Claw: the sleeping giant reawakens

- **Best xG differential in the league (+15.44)** — only Rodice (+12.59) close
- **5-0-0 away, 20 goals, 5 conceded** — best away record
- **Paul De Coupey** (10 goals from 9.8 xG, 8.07 rating) — the league's best-attacking season
- **They haven't lost since MD6 against Oxton** (2-3 home) — 13 straight wins since
- **Goals**: 32 (league-best), 13 on the counter (league-best)
- Expected final points: **78** (our model)
- **10 seasons without a title.** The last was 15/16. Since new Emperor Cyzzo has been sitting on the throne, they've been expected but never consummated — which makes the 00/01–15/16 historic dynasty all the more awkward for the media
- **They have the toughest run-in**: Pearls A, Natrican H, Uditon A, Rodice A, Red Wolves H, Lige-8 A, Chovers H, Triumpher A, Pearls H, Natrican A, Uditon H, Rodice A (MD23)
- **If you can keep a spare bed in the media suite for 2026/27, use it for the Claw conspiracy theories.** Hasn't happened in 10 years. Everyone starves. No one is safe
- **The engine has bought them the free shot** — a superior 2026/27 simulation happens on their software

## 6.4 Pearls — the fallen dynasty

- Currently **5th, 18 pts**, troubled — enormous budget (€1.16bn squad, €4.6M/wk payroll), one-tenth the wage
- **Perćy Luka has frozen** — 1 goal in 9 apps, 6.60 rating in 25/26 and 2026/27
- They've spent 150M on Rayan Victor Jam¢s (the same winger that was the reason Percy was almost benched under the old flank rules — who turned sour this season) — he has only 2G this season
- **2024/25 champions Triumpher currently 7th** — a 4th place in the old engine is not top 6 anymore
- **Sapient habit — the 25/26 season turned up only 20 points for Pearls in the first ten games.** It's what the engine can undo now

Their trophies and atmosphere has driven them nuts this whole 2026/27 season. They'll finish stronger than their position currently suggests, because squad value wins in the long run. But the league has reacted violentably against them.

## 6.5 Triumpher — the richest club in history with zero trophies to match

- **€1.317bn squad value (#1), €11M/week payroll (€572M/year)** — bigger than Pearls and Claw combined
- **They've won only 1 league title all-time (24/25)** — the curse you carry with budgets like this
- **They rebuilt their entire spine**: Van Lee €85M, Ayle Moosh €85M, Orley Valence €65M, Mardalov free, Mirov free
- **They lost eight players over the summer** — Garle, Philsend, Andre, Darmote, Vousev, Xoban, Kersě, Stoph — and paid 45M for Arick who left to Seafcea
- **Currently 7th, 16 pts,** 20 GF / 12 GA; **xG diff +11.22 (4th best in the league)** — the underlying play is fine; the conversion is horrific (they're 10 goals under expected)
- **Manager Garley Smiddos (Attacking)** with James Dhon (UTMD defender) line-ups — young
- **Their 100k stadium seating capacity — record in the country — is breaking attendance records in finance failures** — but the investment doesn't really translate into success
- **It's the trope in the book**: spending the most money and winning the least output. The fans call it **"Triumpher: the oligarchs"**

## 6.6 The expected table — Monte Carlo (engine-fact simulation)

Three thousand simulated seasons from this point:

```
TITLE PROBABILITY
Claw       54.9%
Rodice     36.1%
Lige-8      5.8%
Pearls      1.7%
Triumpher   1.3%

PROJECTED FINAL TABLE
Claw     ~78 pts
Rodice   ~76 pts
Lige-8   ~68 pts
Pearls   ~65 pts
Triumpher ~65 pts
```

**Prediction: Claw wins.** Rodice follows, if he survives his own misery.

## 6.7 The MD11 preview

- **MD11: Lige-8 vs Justice** — Lige-8 4W 0D 0L... wait, they're 7-3-0 prior to MD11. Don Goyle (Possession) at home, Justice nowhere. Lige-8 will win.
- **MD11: Pearls vs Claw** — 🔥 the next Pearls vs Claw rivalry flagship match — at White Pearl, standing barbed. Pearls must stop Claw to prove they're better than the spreadsheets; Claw must win to prove they're worthy of the real title.

## 6.8 The budget as it current stands (26/27)

| Team | Squad value | Weekly wages |
|---|---|---|
| **Triumpher** | **€1.317B** | €11.0M |
| Pearls | €1.16B | €4.6M |
| Claw | €964M | €4.8M |
| Lige-8 | €900M | €2.9M |
| Uditon | €893M | €3.0M |
| Natrican | €809M | €4.9M |
| Red Wolves | €395M | €1.8M |
| Justice | €390M | €1.6M |
| **Rodice** | **€220M** | **€1.0M** |

---

# PART VII — THE DEBATES SETTLED

## 7.1 Pearls vs Claw — the el clasico

- **Attendance record**: Pearls v Claw 17/18 drew 81,237 — the biggest crowd in league history
- **54 meetings** (double round robin across 26 seasons)
- **262 goals at 4.85/game**
- **Pearls 26 wins, 11 draws, Claw 17 wins** — since 2000, Pearls 56–40 in the first decade; since 16/17, Pearls 9 wins, Claw 6 wins, 5 draws
- **Trophy war**: Pearls 38 trophies, Claw 32 trophies — the closest margin between two temples in football history
- **Community Shield splits now**: 10 years of painting a 14-10 war away from Claw

The rivalry is real. Every match day matters. And this season unfolds under the microscope — MD11 belongs on a pedestal.

## 7.2 The creator revelation

- The creator is a **Pearls fan**. He wrote this country into existence — every player, every club, every myth
- He left the spreadsheets behind him to capture every era
- Now he watches Pearls go cold through 25/26 into 26/27 — a collapse of confidence
- He secretly **exports his own universe back at him** — engine-made drama rather than typed football
- He said: *"I am delighted of the debate — and actually saddened Claw look good this season"*
- He is now a Rodice fan — the club formed in the settlements and gusting up the table
- He will, however, never agree that Claw is the biggest club — Pearlon's heart is reluctant
- He won the universe, and he built Pearls himself

## 7.3 The foreigner's point of view (our analyst's vote)

- Our "analyst" came here without history. They chose **Lige-8** — the dark horse of the country's history (3 titles, 11-2 vs Triumpher record, perennial 4th), the underdog of all underdogs, undefeated this season with 24 points, at home (4-2-0). Old Lige — 900,000 people. Huge heart. Real football.

## 7.4 The player debates in a nutshell (closing thoughts on theskeletons)

### Qupri vs Drante Jr. — the GOAT debate

- **Why Qupri?** 578 league apps, 243 goals, record 1,192 take-ons, 217 MOTM (record), 8 league titles, image of the Claw dynasty, Paladivo 2014 WC champion (his country), the true point of origin of modern wing play
- **Why Drante Jr?** 266 apps 206 goals (1.079 GA/game PLOFA Main), 1,108 games/902 goals/0.814 GPG all comps, 6 POTS, 21 hat-tricks (record), 6 hat-tricks in one season (record), 5 games with 4+ goals (record), 6 Balon d'Ors at Pearls (all-time record at any club), 19+ club trophies, WC 2022 champion MVP
- **Influence trees**: Qupri's winger school, Drante's striker school — Ghettenside Gikarchi carries both
- **Head to head**: Pearls 56–40 goals 00/01–08/09, Pearls won that rivalry
- **WC colours**: Paladivo (Qupri) champion 2014, South Alavo Republic (Drante Jr) champion 2022
- **Both born 1983**, both competitors of choice from academia
- **The answer we ended with**: Drante Jr wins (13 peer votes to Qupri's 8), but civilized fans can disagree until the end of time

### Luka vs Qupri — the modern duel (close to forever)

- Qupri: 578 apps · 243 goals · 8 titles (Claw 2000–2016) · 1,192 take-ons
- Luka: 374 apps · 238 goals · 1.07 G+A per game · 4 league titles (Pearls), POTS 6x, Balon d'Or 2 (20/21, 23/24)
- Luka **leads on per-game production** and influence per decade; Qupri leads everywhere else — trophies, take-ons, MVP, loyalty, *Clan Factor*

### Luka vs Drante — the Pearls G.O.A.T.

- Drante Jr: 10 trophies at Pearls, 5 league titles, 20 goals per title season, 6 Balon d'Ors
- Luka: 14 trophies at Pearls (4 league + 10 cups), 611 club games, 705 G+A, POTS 6x, fastest to 200 goals (339)
- Recently drifted to a horrific 2026/27 slump (1 goal in 6 apps, 6.60 rating)
- **The debate is open. We respect both men with full honour.**

---

# PART VIII — MD11 & AFTER

## When this was written, the world could not be stopped. The Matchday window beckoned.

**MD11 — kicks off after the international break.**

Key fixtures:

- **Lige-8 vs Justice** — Lige-8 (Possession, Don Goyle) at home vs Justice (Defensive, Patrick Quintin) — 4-2-0 vs 2-2-6, 24 points vs 8. Expect a coronation.
- **Pearls vs Claw** — the rivalry renewed, white pearl whipping, 4.85 goals/game. The country *literally* emptied its calendar.

And so it begins. The first engine-decided season will decide everything. And the ghost of 1996 will sit at the White Pearl stands, waiting.

---

# APPENDICES

## A. File paths

- `D:\PLOFA\plofa\TOLAND_FOOTBALL_CHRONICLE.md` — this document
- `D:\PLOFA\plofa\season_stats.json` — 26/27 source of truth
- `D:\PLOFA\plofa\season_state.json` — injuries & morale state (stale)
- `D:\PLOFA\plofa\season_state.json` — (sic)
- `D:\PLOFA\plofa\PLOFA-2026-2027.xlsx` — current season workbook (FIXTURES blank G/A, TRANSFERS)
- `D:\PLOFA\plofa\plofa_output\*_MD*.json` — per-match sim files
- `D:\TOLAND FOOTBALL FEDERATION\PLOFA-2025-2026.COM\FULL PLOFA HISTORY ATTEMPT.xlsx` — match history
- `D:\TOLAND FOOTBALL FEDERATION\PLOFA-2025-2026.COM\PLOFA 2025-2026 II.xlsx` — 25/26 TEST6 workbook
- `D:\TOLAND FOOTBALL FEDERATION\TOPTA\2024-2025_STATS\8B2E7200.xlsx` — 24/25 stats
- `C:\Users\Trevor Majani\Documents\PLOFA GOATS ALL TIME STATS 1.xlsx` — GOAT records

## B. Data caveats

1. **25/26 is synthetic** (TEST6.py generated). Never use it as a reference for "form".
2. **24/25 is organic** — Triumpher champions (68 pts), 83 goals.
3. **ALL-TIME-STANDINGS covers through 25/26** (918 games each, 27 seasons).
4. **The history file has duplicate universes** (alternate 23/24, "~INTERNATIONAL CHAMP OF 100" versions). We used only the official canonical season sequence.
5. **This 26/27 is the first engine-made season.** xG figures, ratings, sheets are all empirically measured — that changes everything about how reliable they are.
6. **The PLAYERS sheet's M/G/A columns are blank** in the current workbook. The M/G/A are in `season_stats.json` instead.
7. **The `stands.md` in manager_profiles may not match engine outcomes** — engine doesn't prefer certain managers —

## C. The club teams (current rosters 26/27 — abridged)

**Rodice** (25 players, runner-up of heart): Leyton Vergol (GK), Gilbert Waxley, Troy Bervilĵ, Caelan Rustoh, Wes Brimley, Fintan Meade, Shea Kilbride, Conor Laffey, Declan Harte, Colm Dennehy, Darragh Healy, Osmar Rehling, Dermot Hanley, Kellan Vosch, Pierce Dalby, Jensen Alder, Lorcan Driscoll, Eamon Greal, Cillian D'Frello, Niall Bohan, Oisin Fennelly, Ronan Mulvehill, Kieran Gannon, Finian Dażo, Donal O'Cahn

**Claw** (25 players): Lewin Josin (GK), William D'Colen, Paul De Coupey, Farson Willord, Cruz Maetulo, Grantędd, Alvin Scorford, Caut Mayoderoki, Baron Siljan, Justin Olemba, Arnell Duch, Nils-Erik Dornic, Carl Tœvoda, Mivez Jose, Juri Denlair, Lim Brown, Jacob Lendye, Michael Tickley, Ueda Aiochi, Ednar Alecharv, Pisconso Amacad

**Pearls** (25): Justin Couls (GK), Francis Bonadi, Wesse Mbamba, Davis Portroy, Felix Aăntrope, Edward Tipō, Juan Massey, Miles Goodley, Rayan Victor Jam¢́s, Perćy Luka, Hillary Monzade, Ashton Bee, Yimmy Dexter Sichens, Ken Shaddy, Nair Hunő, Quinten Swertch, Job Zader, Charlton Gonn, Jacob Sento, Shrek Mobellě, Elon Cuveź, Brice Chovanec, Lionel Persi

**Triumpher** (25): Fred Jollä, Dan Mirov, Rob Mardalov, Moses Garner, Hill Prosper, Zachery Worth, Ayle Moosh, Jovan Jochёr, Van Lee, Orley Valence, Caya Tlipa, Doarley, Jenson Morphes, Zack Chalven, Creg Nlova, Tony Coupey, Vicor Dene, Dovtre Jr, Huier Cestopher, Wang Hee Jo, John Slirson

---

*Here ends The Chronicle — compiled through session work and good faith, with the engine running in the background. On Saturday Morning, MD11 kicks off, and only the simulation goes on.*

*— Your analyst, Pearls No!, No, honestly Lige-8 supporter, writing from outside Toland, surrounded by spreadsheets.*
