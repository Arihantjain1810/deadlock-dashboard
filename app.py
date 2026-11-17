"""Deadlock Analysis & Resource Allocation Dashboard (Streamlit).
Run locally:  pip install -r requirements.txt  then  streamlit run app.py
"""
import copy
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Deadlock Analysis Dashboard", layout="wide")

# ------------------------------------------------------------------ core logic
CLASSIC = {
    "alloc": [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]],
    "max": [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]],
    "avail": [3, 3, 2],
}
DETECT_PRESETS = {
    "Multiple-instance: deadlock (P0, P1, P2)": {
        "alloc": [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
        "req": [[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 0]],
        "avail": [0, 0, 0], "single": False},
    "Multiple-instance: cycle but no deadlock": {
        "alloc": [[1, 0], [0, 1], [1, 0]],
        "req": [[0, 1], [1, 0], [0, 0]],
        "avail": [0, 0], "single": False},
    "Single-instance: cycle means deadlock": {
        "alloc": [[1, 0], [0, 1]],
        "req": [[0, 1], [1, 0]],
        "avail": [0, 0], "single": True},
}
LABEL = {"safe": "SAFE", "unsafe": "UNSAFE", "waiting": "WAITING", "deadlock": "DEADLOCK", "invalid": "INVALID INPUT"}


def le(a, b): return all(x <= y for x, y in zip(a, b))
def add(a, b): return [x + y for x, y in zip(a, b)]
def sub(a, b): return [x - y for x, y in zip(a, b)]
def P(i): return f"P{i}"
def bad(v): return v != v or v is None or float(v) != int(v) or v < 0


def frame(mat, cols=None, rows=None):
    m = len(mat[0])
    return pd.DataFrame(mat, columns=cols or [f"R{j}" for j in range(m)],
                        index=rows or [P(i) for i in range(len(mat))])


def clean(df):
    return [[(x if x != x else int(x)) for x in row] for row in df.values.tolist()]


def validate(S):
    errs = []
    for j, v in enumerate(S["avail"]):
        if bad(v): errs.append(f"Available R{j} must be a non-negative whole number.")
    for i, row in enumerate(S["alloc"]):
        for j, v in enumerate(row):
            mx = S["max"][i][j]
            if bad(v): errs.append(f"Allocation {P(i)},R{j} must be a non-negative whole number.")
            elif bad(mx): errs.append(f"Maximum {P(i)},R{j} must be a non-negative whole number.")
            elif v > mx: errs.append(f"{P(i)} holds {int(v)} of R{j}, more than its declared maximum {int(mx)}.")
    return errs


def need_of(S): return [sub(S["max"][i], S["alloc"][i]) for i in range(len(S["alloc"]))]


def safety(alloc, need, avail):
    work, fin, seq, steps = list(avail), [False] * len(alloc), [], []
    while True:
        pick = next((i for i in range(len(alloc)) if not fin[i] and le(need[i], work)), None)
        if pick is None: break
        before = list(work)
        work = add(work, alloc[pick]); fin[pick] = True; seq.append(pick)
        steps.append({"Process": P(pick), "Need": str(need[pick]), "Work before": str(before),
                      "Work after (+Allocation)": str(work)})
    return {"safe": all(fin), "seq": seq, "steps": steps, "stuck": [i for i, f in enumerate(fin) if not f]}


def check_request(S, p, req):
    n, m = len(S["alloc"]), len(S["avail"])
    if not isinstance(p, int) or p < 0 or p >= n:
        return {"status": "invalid", "msg": f"Process P{p} does not exist (valid IDs: P0 to P{n-1})."}
    if len(req) != m or any(not isinstance(v, int) for v in req):
        return {"status": "invalid", "msg": f"A request needs {m} whole numbers, one per resource type."}
    if any(v < 0 for v in req):
        return {"status": "invalid", "msg": "The request has a negative value. A process cannot request negative units."}
    need = need_of(S)
    if not le(req, need[p]):
        return {"status": "invalid", "msg": f"Request {req} exceeds {P(p)}'s remaining Need {need[p]}: it breaks the declared maximum claim."}
    if not le(req, S["avail"]):
        return {"status": "waiting", "msg": f"Request {req} is within Need {need[p]} but above Available {S['avail']}. {P(p)} must wait; the allocation is unchanged. This is not an error and not a permanent denial."}
    T = {"alloc": copy.deepcopy(S["alloc"]), "max": S["max"], "avail": sub(S["avail"], req)}
    T["alloc"][p] = add(T["alloc"][p], req)
    r = safety(T["alloc"], need_of(T), T["avail"])
    if r["safe"]:
        return {"status": "safe", "msg": f"Pretend-allocating {req} to {P(p)} leaves a safe state, so the request is granted.", "r": r, "new": T}
    return {"status": "unsafe", "msg": f"Pretend-allocating {req} to {P(p)} leaves no complete safe sequence (stuck: {', '.join(P(i) for i in r['stuck'])}). Request denied; the original state is restored.", "r": r}


def detect(alloc, req, avail):
    work, fin = list(avail), [all(v == 0 for v in row) for row in alloc]
    steps, changed = [], True
    while changed:
        changed = False
        for i in range(len(alloc)):
            if not fin[i] and le(req[i], work):
                before = list(work); work = add(work, alloc[i]); fin[i] = True; changed = True
                steps.append({"Process": P(i), "Request": str(req[i]), "Work before": str(before), "Work after": str(work)})
    return {"dead": [i for i, f in enumerate(fin) if not f], "steps": steps}


def simulate(policy):
    plans = [[0, 1], [1, 0]]
    plans = [sorted(p) if policy == "order" else p for p in plans]
    st_ = [{"idx": 0, "done": False, "ready": False} for _ in plans]
    own, log, t, dead = [-1, -1], [], 0, False
    while any(not s["done"] for s in st_) and t < 20:
        t += 1; prog = False
        for i, plan in enumerate(plans):
            s = st_[i]
            if s["done"]: continue
            if s["ready"]:
                for r in plan: own[r] = -1
                s["done"] = prog = True; log.append(f"t{t}: {P(i)} finishes, releases {', '.join(f'R{r}' for r in plan)}"); continue
            if policy == "all":
                if all(own[r] == -1 for r in plan):
                    for r in plan: own[r] = i
                    s["ready"] = prog = True; log.append(f"t{t}: {P(i)} takes all of {', '.join(f'R{r}' for r in plan)} at once")
                else: log.append(f"t{t}: {P(i)} holds nothing and waits")
                continue
            r = plan[s["idx"]]
            if own[r] == -1:
                own[r] = i; s["idx"] += 1; prog = True; log.append(f"t{t}: {P(i)} acquires R{r}")
                if s["idx"] == len(plan): s["ready"] = True
            else:
                log.append(f"t{t}: {P(i)} blocked on R{r} (held by {P(own[r])})")
        if not prog: dead = True; break
    return {"log": log, "dead": dead, "ticks": t, "plans": plans}


def note(msg): st.session_state.last = msg


def status_box(status, text):
    {"safe": st.success, "unsafe": st.warning, "waiting": st.info, "deadlock": st.error, "invalid": st.error}[status](f"**{LABEL[status]}**: {text}")


# ------------------------------------------------------------------ state
if "cfg" not in st.session_state: st.session_state.cfg = copy.deepcopy(CLASSIC)
if "last" not in st.session_state: st.session_state.last = "No simulation yet"
if "ver" not in st.session_state: st.session_state.ver = 0
cfg = st.session_state.cfg

MODULES = ["Overview Dashboard", "System Configuration", "Banker's Algorithm", "Resource Request Simulator",
           "Deadlock Detection", "Deadlock Prevention", "Test Lab", "Concepts and Learning", "Responsible AI Usage"]
with st.sidebar:
    page = st.radio("Simulation modules", MODULES)
    st.divider()
    st.caption("Latest simulation outcome")
    st.info(st.session_state.last)
    if st.button("Reset to classic example"):
        st.session_state.cfg = copy.deepcopy(CLASSIC); st.session_state.ver += 1; note("Reset to classic example"); st.rerun()

st.title("Deadlock Analysis & Resource Allocation")
st.caption("Operating Systems simulation laboratory")

# ------------------------------------------------------------------ pages
if page == "Overview Dashboard":
    st.header("Overview")
    c = st.columns(3)
    c[0].metric("Processes", len(cfg["alloc"])); c[1].metric("Resource types", len(cfg["avail"])); c[2].metric("Available", str(cfg["avail"]))
    errs = validate(cfg)
    if errs: status_box("invalid", "; ".join(errs))
    else:
        r = safety(cfg["alloc"], need_of(cfg), cfg["avail"])
        if r["safe"]: status_box("safe", "Current state is safe. Sequence: " + " → ".join(P(i) for i in r["seq"]))
        else: status_box("unsafe", f"Current state is unsafe (stuck: {', '.join(P(i) for i in r['stuck'])}). Unsafe is not the same as deadlocked.")
    a, b, d = st.columns(3)
    a.subheader("Allocation"); a.dataframe(frame(cfg["alloc"]))
    b.subheader("Maximum"); b.dataframe(frame(cfg["max"]))
    if not errs: d.subheader("Need"); d.dataframe(frame(need_of(cfg)))
    st.markdown("**Status labels used here:** SAFE (a safe sequence exists), UNSAFE (no safe sequence, but not necessarily deadlocked), WAITING (request within Need but above Available), DEADLOCK (processes blocked on each other), INVALID INPUT (malformed values or a claim above the declared maximum).")

elif page == "System Configuration":
    st.header("System configuration")
    c1, c2, c3 = st.columns([1, 1, 2])
    n = c1.number_input("Processes", 1, 8, len(cfg["alloc"])); m = c2.number_input("Resource types", 1, 6, len(cfg["avail"]))
    if c3.button("Apply new size (clears matrices)"):
        st.session_state.cfg = {"alloc": [[0] * m for _ in range(n)], "max": [[0] * m for _ in range(n)], "avail": [0] * m}
        st.session_state.ver += 1; st.rerun()
    v = st.session_state.ver
    a = st.data_editor(frame(cfg["alloc"]), key=f"alloc{v}"); mx = st.data_editor(frame(cfg["max"]), key=f"max{v}")
    av = st.data_editor(frame([cfg["avail"]], rows=["Available"]), key=f"av{v}")
    if st.button("Save configuration", type="primary"):
        new = {"alloc": clean(a), "max": clean(mx), "avail": clean(av)[0]}
        errs = validate(new)
        if errs: status_box("invalid", " ".join(errs)); note("Invalid configuration rejected")
        else: st.session_state.cfg = new; st.success("Configuration saved."); note("Configuration saved")

elif page == "Banker's Algorithm":
    st.header("Banker's Algorithm: safety check")
    errs = validate(cfg)
    if errs: status_box("invalid", " ".join(errs))
    else:
        need = need_of(cfg)
        st.subheader("Need = Maximum − Allocation"); st.dataframe(frame(need))
        if st.button("Run safety check", type="primary"):
            r = safety(cfg["alloc"], need, cfg["avail"])
            if r["safe"]:
                status_box("safe", "Safe sequence " + " → ".join(P(i) for i in r["seq"]) + ". Each process in turn has Need ≤ Work, so it can finish and release what it holds.")
                note("SAFE: " + " → ".join(P(i) for i in r["seq"]))
            else:
                status_box("unsafe", f"Only {len(r['seq'])} process(es) can finish. Stuck: {', '.join(P(i) for i in r['stuck'])}. Unsafe is not deadlock: nobody is blocked yet, but completion cannot be guaranteed.")
                note("UNSAFE state")
            if r["steps"]: st.dataframe(pd.DataFrame(r["steps"]), hide_index=True)

elif page == "Resource Request Simulator":
    st.header("Resource request simulator")
    st.caption("Banker's avoidance: grant only if the state after granting is safe. It does not detect existing deadlock.")
    EX = {"Custom": ("1", "1,0,2"), "P1 [1,0,2] (grant)": ("1", "1,0,2"), "P0 [4,0,0] (exceeds Available)": ("0", "4,0,0"),
          "P4 [3,3,0] (unsafe)": ("4", "3,3,0"), "P3 [1,0,0] (exceeds Need)": ("3", "1,0,0"),
          "P9 [1,0,0] (bad process ID)": ("9", "1,0,0"), "P1 [-1,0,0] (negative)": ("1", "-1,0,0")}
    ex = st.selectbox("Quick example (written for the classic sample)", list(EX))
    c1, c2 = st.columns(2)
    pid = c1.text_input("Process ID", EX[ex][0], key=f"pid{ex}"); rv = c2.text_input("Request vector (comma separated)", EX[ex][1], key=f"rv{ex}")
    if st.button("Submit request", type="primary"):
        errs = validate(cfg)
        if errs: status_box("invalid", " ".join(errs))
        else:
            try: p = int(pid.strip().upper().lstrip("P"))
            except ValueError: p = None
            try: req = [int(x) for x in rv.replace(",", " ").split()]
            except ValueError: req = [0.5]
            r = check_request(cfg, p if p is not None else -1, req)
            if p is None: r = {"status": "invalid", "msg": "Process ID must be a number such as 1 or P1."}
            if r["status"] == "safe":
                status_box("safe", "Request granted. " + r["msg"] + " New safe sequence: " + " → ".join(P(i) for i in r["r"]["seq"]))
                st.dataframe(pd.DataFrame(r["r"]["steps"]), hide_index=True)
                st.session_state.cfg = {"alloc": r["new"]["alloc"], "max": r["new"]["max"], "avail": r["new"]["avail"]}; st.session_state.ver += 1
                st.caption("The configuration now holds the new state.")
            elif r["status"] == "unsafe":
                status_box("unsafe", r["msg"] + " This is unsafe, not deadlocked: no process is waiting forever yet.")
                if r["r"]["steps"]: st.dataframe(pd.DataFrame(r["r"]["steps"]), hide_index=True)
            else: status_box(r["status"], r["msg"])
            note(f"{LABEL[r['status']]}: P{p} {req}")

elif page == "Deadlock Detection":
    st.header("Deadlock detection and recovery")
    key = st.selectbox("Scenario", list(DETECT_PRESETS))
    if st.session_state.get("det_key") != key:
        st.session_state.det = copy.deepcopy(DETECT_PRESETS[key]); st.session_state.det_key = key; st.session_state.dver = st.session_state.get("dver", 0) + 1
    D, dv = st.session_state.det, st.session_state.dver
    tot = [D["avail"][j] + sum(r[j] for r in D["alloc"]) for j in range(len(D["avail"]))]
    st.info(("Resource model: **single instance**" if D["single"] else "Resource model: **multiple instances**") + f" (total per type {tot}). " +
            ("A cycle is conclusive here." if D["single"] else "A cycle alone is not conclusive, so the detection algorithm is used."))
    c = st.columns(3)
    with c[0]: st.caption("Allocation"); a = st.data_editor(frame(D["alloc"]), key=f"da{dv}")
    with c[1]: st.caption("Request (outstanding)"); q = st.data_editor(frame(D["req"]), key=f"dq{dv}")
    with c[2]: st.caption("Available"); av = st.data_editor(frame([D["avail"]], rows=["Avail"]), key=f"dv{dv}")
    S = {"alloc": clean(a), "req": clean(q), "avail": clean(av)[0]}
    r = detect(S["alloc"], S["req"], S["avail"])
    dot = ["digraph G { rankdir=LR; node [fontsize=11];"]
    for i in range(len(S["alloc"])):
        dot.append(f'P{i} [shape=circle,{"color=red,style=filled,fillcolor=mistyrose" if i in r["dead"] else ""}];')
    for j in range(len(S["avail"])): dot.append(f"R{j} [shape=box];")
    for i, row in enumerate(S["alloc"]):
        for j, v in enumerate(row):
            if v: dot.append(f'R{j} -> P{i} [label="{v}",color=blue];')
    for i, row in enumerate(S["req"]):
        for j, v in enumerate(row):
            if v: dot.append(f'P{i} -> R{j} [label="{v}",style=dashed,color=orange];')
    st.graphviz_chart("\n".join(dot) + "}")
    st.caption("Blue solid: resource held by process. Orange dashed: process waiting for resource.")
    if r["dead"]:
        status_box("deadlock", f"Deadlocked processes: {', '.join(P(i) for i in r['dead'])}. Each needs resources only another deadlocked process holds.")
        note(f"DEADLOCK: {', '.join(P(i) for i in r['dead'])}")
    else:
        status_box("safe", "No deadlock. " + ("The graph has a cycle, but a spare instance lets a process finish and release resources." if "cycle" in key else "All requests can eventually be satisfied."))
        note("No deadlock detected")
    if r["steps"]: st.dataframe(pd.DataFrame(r["steps"]), hide_index=True)
    st.subheader("Recovery")
    vic = st.selectbox("Process to terminate", range(len(S["alloc"])), format_func=P)
    if st.button("Terminate and re-run detection"):
        rel = S["alloc"][vic]
        S["avail"] = add(S["avail"], rel); S["alloc"][vic] = [0] * len(rel); S["req"][vic] = [0] * len(rel)
        st.session_state.det.update(alloc=S["alloc"], req=S["req"], avail=S["avail"]); st.session_state.dver += 1
        after = detect(S["alloc"], S["req"], S["avail"])
        note(f"Terminated {P(vic)}; released {rel}")
        st.success(f"Terminated {P(vic)} and released {rel}. Available is now {S['avail']}. " +
                   ("Still deadlocked: " + ", ".join(P(i) for i in after["dead"]) if after["dead"] else "The remaining processes can now finish."))

elif page == "Deadlock Prevention":
    st.header("Deadlock prevention")
    st.write("Two processes, two single-instance resources. P0 needs R0 then R1; P1 needs R1 then R0.")
    cols = st.columns(3)
    for col, (k, title) in zip(cols, [("none", "Original: no policy"), ("order", "Fixed order R0 then R1"), ("all", "Request all at once")]):
        s = simulate(k)
        with col:
            st.subheader(title); st.code("\n".join(s["log"]), language=None)
            if s["dead"]: st.error("DEADLOCK: circular wait reached")
            else: st.success(f"Completed in {s['ticks']} ticks")
    st.markdown("**Trade-offs.** Ordering removes circular wait but forces a rigid acquisition order. Requesting everything at once removes hold and wait but leaves resources idle and can cause starvation. Both reduce concurrency compared with the unrestricted policy.")
    note("Prevention comparison viewed")

elif page == "Test Lab":
    st.header("Reproducible assessment test lab: TC1 to TC6")
    st.write("Each result is computed by running the project's algorithms on fresh scenario state.")
    def run_tests():
        B, rows = copy.deepcopy(CLASSIC), []
        t1 = safety(B["alloc"], need_of(B), B["avail"])
        rows.append(("TC1", "Classic safe state", "SAFE; all processes finish", "SAFE: " + " → ".join(P(i) for i in t1["seq"]) if t1["safe"] else "UNSAFE", t1["safe"], "Each picked process has Need ≤ Work; Work grows as it releases its allocation."))
        t2 = check_request(B, 0, [4, 0, 0])
        rows.append(("TC2", "Request exceeds Available (P0 [4,0,0])", "WAITING; state unchanged", t2["status"].upper(), t2["status"] == "waiting", "Within Need [7,4,3] but 4 > 3 in R0. Valid request, not invalid, not permanently denied."))
        snap = copy.deepcopy(B); t3 = check_request(B, 4, [3, 3, 0])
        rows.append(("TC3", "Tentative request unsafe (P4 [3,3,0])", "DENIED; exact rollback", f"{t3['status'].upper()}; unchanged={B == snap}", t3["status"] == "unsafe" and B == snap, "Available becomes [0,0,2] and no Need fits. Denied and restored. Unsafe, not deadlocked."))
        d = DETECT_PRESETS["Multiple-instance: deadlock (P0, P1, P2)"]; t4 = detect(d["alloc"], d["req"], d["avail"])
        rows.append(("TC4", "Independent detection (multi-instance)", "Detect P0, P1, P2", f"Detected {[P(i) for i in t4['dead']]}", t4["dead"] == [0, 1, 2], "Detection uses outstanding Request, not Banker's Need."))
        al, rq, av = copy.deepcopy(d["alloc"]), copy.deepcopy(d["req"]), list(d["avail"]); rel = al[0]; av = add(av, rel); al[0] = [0, 0, 0]; rq[0] = [0, 0, 0]
        t4b = detect(al, rq, av)
        rows.append(("TC4 recovery", "Terminate P0 and release allocation", "No remaining deadlocked process", "No deadlock identified" if not t4b["dead"] else "Still deadlocked", not t4b["dead"], f"Released {rel}; Available is {av}."))
        s0, s1, s2 = simulate("none"), simulate("order"), simulate("all")
        rows.append(("TC5", "Prevention policies", "Original deadlocks; both policies complete", f"none: {'deadlock' if s0['dead'] else 'ok'}; order: {'deadlock' if s1['dead'] else str(s1['ticks'])+' ticks'}; all-at-once: {'deadlock' if s2['dead'] else str(s2['ticks'])+' ticks'}", s0["dead"] and not s1["dead"] and not s2["dead"], "Ordering breaks circular wait; all-at-once breaks hold and wait."))
        z = copy.deepcopy(B); z["avail"] = [0, 0, 0]; rz = safety(z["alloc"], need_of(z), z["avail"])
        rows.append(("TC6", "Zero Available", "UNSAFE", "UNSAFE" if not rz["safe"] else "SAFE", not rz["safe"], "Positive Need cannot be satisfied by zero Work."))
        zn = copy.deepcopy(B); zn["max"][2] = list(zn["alloc"][2])
        rows.append(("TC6", "Zero-Need process requests more", "INVALID", check_request(zn, 2, [1, 0, 0])["status"].upper(), check_request(zn, 2, [1, 0, 0])["status"] == "invalid", "Request exceeds remaining Need of zero."))
        rows.append(("TC6", "Zero-vector request", "SAFE (no change)", check_request(B, 1, [0, 0, 0])["status"].upper(), check_request(B, 1, [0, 0, 0])["status"] == "safe", "Valid boundary case."))
        rows.append(("TC6", "Unknown process P9", "INVALID", check_request(B, 9, [1, 0, 0])["status"].upper(), check_request(B, 9, [1, 0, 0])["status"] == "invalid", "No such process."))
        rows.append(("TC6", "Negative request", "INVALID", check_request(B, 1, [-1, 0, 0])["status"].upper(), check_request(B, 1, [-1, 0, 0])["status"] == "invalid", "Negative units are rejected."))
        bd = copy.deepcopy(B); bd["alloc"][0][0] = 8
        rows.append(("TC6", "Allocation above Maximum", "INVALID", "INVALID" if validate(bd) else "accepted", bool(validate(bd)), "State validation rejects Allocation > Maximum."))
        return pd.DataFrame([{"Test ID": a, "Scenario": b, "Expected Result": c, "Actual Result": d_, "Pass/Fail": "PASS" if e else "FAIL", "Analysis": f} for a, b, c, d_, e, f in rows])
    if st.button("Rerun all assessment scenarios", type="primary") or True:
        df = run_tests(); st.dataframe(df, hide_index=True, use_container_width=True)
        note(f"Test Lab: {(df['Pass/Fail'] == 'PASS').sum()}/{len(df)} passed")

elif page == "Concepts and Learning":
    st.header("Concepts and learning")
    for t, body in [("Prevention", "Design rules so one of the four necessary conditions (mutual exclusion, hold and wait, no preemption, circular wait) cannot hold. Cost: lower utilization and concurrency."),
                    ("Avoidance", "Decide each request at run time. The Banker's Algorithm grants a request only if the resulting state is safe. It needs maximum claims in advance and does not detect existing deadlock."),
                    ("Detection", "Let processes run, then inspect Allocation, Request and Available. A cycle in a resource-allocation graph proves deadlock only for single-instance resources."),
                    ("Recovery", "Terminate or preempt to break the deadlock. Released resources return to Available so others can progress.")]:
        with st.expander(t, expanded=True): st.write(body)
    st.info("Unsafe is not deadlocked. Unsafe means no safe sequence exists; deadlock means processes are actually blocked on each other.")

elif page == "Responsible AI Usage":
    st.header("Responsible AI usage")
    st.write("List the AI tools your group used, what each one produced, and how you checked it. Add a row for every tool.")
    df = pd.DataFrame({"AI tool used": ["", "", ""], "Purpose": ["Dashboard design", "Algorithm/code assistance", "Test-case generation or debugging"],
                       "Output generated": ["", "", ""], "Student verification": ["", "", ""]})
    out = st.data_editor(df, num_rows="dynamic", hide_index=True, use_container_width=True)
    st.download_button("Download record as CSV", out.to_csv(index=False), "ai_usage_record.csv")
