import json, glob, math, os, collections, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
MATCHES = glob.glob("plofa_output/*/*.json")
SHOT = {"SHOT_ON_TARGET","SHOT_OFF_TARGET","SHOT_BLOCKED","GOAL","HIT_WOODWORK","FREEKICK_DIRECT","PENALTY_SCORED","PENALTY_MISSED"}
def nx(x, ar): return x if ar else 105.0 - x
rows=[]
for path in sorted(MATCHES):
    d=json.load(open(path,encoding="utf-8",errors="replace"))
    m=d.get("match",{})
    home=(m.get("home_team") or m.get("home")) if isinstance(m,dict) else m[0]
    away=(m.get("away_team") or m.get("away")) if isinstance(m,dict) else m[1]
    tl=d.get("timeline") or []
    st=collections.defaultdict(lambda: collections.defaultdict(int))
    sx=collections.defaultdict(float); sc=collections.defaultdict(int)
    pl=collections.defaultdict(float); pn=collections.defaultdict(int)
    ftc=collections.defaultdict(int); ohc=collections.defaultdict(int)
    corner=collections.defaultdict(int); flank=collections.defaultdict(int); cent=collections.defaultdict(int)
    for e in tl:
        if not isinstance(e,dict): continue
        kind=e.get("kind") or e.get("type"); team=e.get("team")
        if team not in (home,away): continue
        ar=(team==home); x=e.get("x",0) or 0; y=e.get("y",0) or 0
        ex=e.get("end_x"); ey=e.get("end_y"); nkx=nx(x,ar)
        if kind=="pass":
            st[team]["passes_intent"]+=1
            if e.get("outcome") is True:
                st[team]["passes_comp"]+=1
                if ex is not None and ey is not None:
                    pl[team]+=math.hypot(ex-x,ey-y); pn[team]+=1
                nkex=nx(ex if ex is not None else x,ar)
                if nkex-nkx>=9.0: st[team]["progressive"]+=1
                if nkex>=70: st[team]["ft3"]+=1
                if nkex>=88.5: st[team]["box"]+=1
        elif kind=="carry":
            st[team]["carries"]+=1
            if e.get("outcome") is True and ex is not None:
                nkex=nx(ex,ar)
                if nkex>=70: ftc[team]+=1
                if nkex>=52.5: ohc[team]+=1
        elif kind in ("cross","FREEKICK_CROSS"):
            st[team]["crosses"]+=1
            if e.get("outcome") is True: st[team]["crosses_comp"]+=1
        elif kind=="through": st[team]["through"]+=1
        elif kind=="dribble":
            st[team]["drb"]+=1
            if e.get("outcome") is True: st[team]["drb_succ"]+=1
        elif kind=="chance": st[team]["chances"]+=1
        elif kind in SHOT:
            st[team]["shots"]+=1
            if e.get("outcome") is True: st[team]["sot"]+=1
            sx[team]+=nx(ex if (ex is not None and ex!=0) else x,ar); sc[team]+=1
        elif kind in ("CORNER_TAKEN","CORNER_WON"): corner[team]+=1
        if kind in ("pass","carry","cross","through","dribble","BALL_RECEIPT"):
            if y<20 or y>48: flank[team]+=1
            else: cent[team]+=1
    seqs=d.get("sequences") or {}
    scnt=collections.defaultdict(int)
    if isinstance(seqs,dict):
        for k,s in seqs.items():
            t=(s.get("Team") or s.get("team")) if isinstance(s,dict) else None
            if t in (home,away): scnt[t]+=1
    for t in (home,away):
        rows.append(dict(team=t,
            pin=st[t]["passes_intent"], pc=st[t]["passes_comp"],
            cp=round(100*st[t]["passes_comp"]/max(1,st[t]["passes_intent"]),1),
            apl=round(pl[t]/max(1,pn[t]),1), prog=st[t]["progressive"],
            ft3=st[t]["ft3"], box=st[t]["box"], carry=st[t]["carries"],
            ftc=ftc[t], ohc=ohc[t], cross=st[t]["crosses"], crossc=st[t]["crosses_comp"],
            thr=st[t]["through"], drb=st[t]["drb"], drbs=st[t]["drb_succ"],
            ch=st[t]["chances"], shots=st[t]["shots"], sot=st[t]["sot"],
            sx=round(sx[t]/max(1,sc[t]),1), corn=corner[t], seq=scnt[t],
            flank=round(100*flank[t]/max(1,flank[t]+cent[t]),1)))
cols=["pin","pc","cp","apl","prog","ft3","box","carry","ftc","ohc","cross","crossc","thr","drb","drbs","ch","shots","sot","sx","corn","seq","flank"]
agg={c: sum(r[c] for r in rows)/max(1,len(rows)) for c in cols}
print("="*110)
print(f"Across {len(rows)} team-performances ({len(MATCHES)} matches)")
print("="*110)
print(f"{'metric':<14}{'avg':>9}")
NAMES={"pin":"passes intent","pc":"passes comp","cp":"comp%","apl":"avg pass len","prog":"progressive","ft3":"passes into final 3rd","box":"passes into box","carry":"carries","ftc":"final-3rd carries","ohc":"opp-half carries","cross":"crosses","crossc":"crosses comp","thr":"through balls","drb":"dribbles","drbs":"dribble succ","ch":"chances","shots":"shots","sot":"shots on target","sx":"avg shot x","corn":"corners","seq":"possession seqs","flank":"flank touch %"}
for c in cols: print(f"{NAMES[c]:<20}{agg[c]:>9.1f}")
print()
print("team | comp% | prog | ft3 | box | cross | thr | drb | shots | SOT | avgSX | corn | seq | flank%")
for r in rows:
    print(f"{r['team'][:16]:<16} | {r['cp']:>4} | {r['prog']:>4} | {r['ft3']:>3} | {r['box']:>3} | {r['cross']:>3} | {r['thr']:>3} | {r['drb']:>3} | {r['shots']:>4} | {r['sot']:>3} | {r['sx']:>5} | {r['corn']:>3} | {r['seq']:>3} | {r['flank']:>4}")
