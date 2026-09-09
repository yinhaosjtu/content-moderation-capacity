import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB


class ModelData(object):
    def __init__(self, inst):
        dim = inst["dimensions"]
        self.nL = int(dim["n_languages"])
        self.nC = int(dim["n_categories"])
        self.nK = int(dim["n_tiers"])
        self.nD = int(dim["n_days"])
        self.nP = int(dim["n_operating_points"])
        self.nW = int(dim["n_scenarios"])

        sets = inst["sets"]
        self.k_min = [int(v) for v in sets["k_min"]]
        self.C_k = {k: [int(c) for c in sets["C_k"][str(k)]] for k in range(1, self.nK + 1)}
        self.A_c = {c: [(str(a[0]), int(a[1])) for a in sets["A_c"][str(c)]] for c in range(self.nC)}
        self.C_res = [int(c) for c in sets["C_res"]]
        self.C_fc = set(int(c) for c in sets["C_fc"])
        self.C_fo = set(int(c) for c in sets["C_fo"])
        self.C_free = [c for c in range(self.nC) if c not in self.C_fc and c not in self.C_fo]

        det = inst["detection_coefficients"]
        self.phi_esc = np.asarray(det["phi_esc"], dtype=float)
        self.phi_mid = np.asarray(det["phi_mid"], dtype=float)
        self.phi_blk = np.asarray(det["phi_blk"], dtype=float)
        self.phi_pas = np.asarray(det["phi_pas"], dtype=float)
        self.e_fn = np.asarray(det["e_fn"], dtype=float)
        self.e_fp = np.asarray(det["e_fp"], dtype=float)
        self.pi_hat = np.asarray(det["pi_hat"], dtype=float)
        self.p_ref = int(inst["operating_points"]["p_ref"])

        vol = inst["volume"]
        self.a_bar = np.asarray(vol["a_bar"], dtype=float)
        self.psi = np.asarray(vol["psi"], dtype=float)

        wk = inst["workload"]
        self.H = float(wk["H"])
        self.theta_in = np.asarray(wk["theta_in"], dtype=float)
        self.theta_out = np.asarray(wk["theta_out"], dtype=float)
        self.theta_qc = np.asarray(wk["theta_qc"], dtype=float)
        self.o_bar = np.asarray(wk["o_bar"], dtype=float)
        self.acc_in = np.asarray(wk["accuracy_in"], dtype=float)
        self.acc_out = np.asarray(wk["accuracy_out"], dtype=float)

        insp = inst["inspection"]
        self.vs_blk = np.asarray(insp["varsigma_blk"], dtype=float)
        self.vs_pas = np.asarray(insp["varsigma_pas"], dtype=float)
        self.vs_adj_in = np.asarray(insp["varsigma_adj_in"], dtype=float)
        self.vs_adj_out = np.asarray(insp["varsigma_adj_out"], dtype=float)

        comp = inst["compliance"]
        self.epsilon_lng = float(comp["epsilon_lng"])
        self.eta_cell = np.asarray(comp["eta_cell"], dtype=float)
        self.varpi_lng = float(comp["varpi_lng"])
        self.varpi_cell = np.asarray(comp["varpi_cell"], dtype=float)
        self.varpi_qc = float(comp["varpi_qc"])

        cost = inst["cost"]
        self.c_hr = np.asarray(cost["c_hr"], dtype=float)
        self.c_ot = np.asarray(cost["c_ot"], dtype=float)
        self.c_out = np.asarray(cost["c_out"], dtype=float)
        self.c_ml = float(cost["c_ml"])
        self.c_fn = np.asarray(cost["c_fn"], dtype=float)
        self.c_fp = np.asarray(cost["c_fp"], dtype=float)
        self.n_bar = np.asarray(cost["n_bar"], dtype=float)
        self.B_bar = float(cost["B_bar"])

        unc = inst["uncertainty"]
        self.alpha = float(unc["alpha"])
        self.gamma = float(unc["gamma"])

        sc = inst["scenarios"]
        self.prob = np.asarray(sc["prob"], dtype=float)
        self.Z = np.asarray(sc["Z"], dtype=float)
        self.gidx = np.asarray(sc["delta_index"], dtype=int)
        self.v_bar = np.asarray(sc["v_bar"], dtype=float)
        self.U_bar = np.asarray(inst["derived"]["U_bar"], dtype=float)

    def handling(self, l, c, r, k):
        if r == "in":
            return float(self.theta_in[l][c][k - 1])
        return float(self.theta_out[l][c])

    def accuracy(self, l, c, r, k):
        if r == "in":
            return float(self.acc_in[l][c][k - 1])
        return float(self.acc_out[l][c])

    def audit_rate(self, c, r):
        if r == "in":
            return float(self.vs_adj_in[c])
        return float(self.vs_adj_out[c])


LAM_CONTINUOUS = True


def build_model(data, env, keep_blocks=False, lock_pref=False):
    model = gp.Model(env=env)
    blocks = {}

    nL, nC, nK, nD, nP, nW = data.nL, data.nC, data.nK, data.nD, data.nP, data.nW

    lcp_all = [(l, c, p) for l in range(nL) for c in range(nC) for p in range(nP)]
    h_index = [(l, c, p, r, k) for (l, c, p) in lcp_all for (r, k) in data.A_c[c]]
    up_index = [(l, c, p) for (l, c, p) in lcp_all if c not in data.C_fc]
    um_index = [(l, c, p) for (l, c, p) in lcp_all if c not in data.C_fo]
    cell_index = [(l, c) for l in range(nL) for c in data.C_res]
    out_by_pool = {l: [(c, p) for c in range(nC) if data.k_min[c] == 1 for p in range(nP)]
                   for l in range(nL)}

    n = model.addVars(nL, nK, vtype=GRB.INTEGER, lb=0.0)
    y = model.addVars(nL, nC, vtype=GRB.BINARY)
    xi = model.addVar(lb=0.0)
    wtail = model.addVars(nW, lb=0.0)
    jcost = model.addVars(nD, nW, lb=0.0)

    for l in range(nL):
        for k in range(nK):
            n[l, k].UB = float(data.n_bar[l][k])
        for c in range(nC):
            if c in data.C_fc:
                y[l, c].LB = 1.0
            elif c in data.C_fo:
                y[l, c].UB = 0.0

    commitment = gp.quicksum(float(data.c_hr[l][k]) * n[l, k]
                             for l in range(nL) for k in range(nK))
    model.addConstr(commitment <= data.B_bar)

    for w in range(nW):
        model.addConstr(wtail[w] >= gp.quicksum(jcost[d, w] for d in range(nD)) - xi)

    expected = gp.quicksum(float(data.prob[w]) * jcost[d, w]
                           for d in range(nD) for w in range(nW))
    tail = xi + (1.0 / (1.0 - data.alpha)) * gp.quicksum(
        float(data.prob[w]) * wtail[w] for w in range(nW))
    model.setObjective(commitment + (1.0 - data.gamma) * expected + data.gamma * tail,
                       GRB.MINIMIZE)

    c_fn = [float(v) for v in data.c_fn]
    c_fp = [float(v) for v in data.c_fp]
    c_out = [float(v) for v in data.c_out]
    c_ot = data.c_ot.tolist()
    theta_qc = data.theta_qc.tolist()
    vs_blk = [float(v) for v in data.vs_blk]
    vs_pas = [float(v) for v in data.vs_pas]
    varpi_cell = [float(v) for v in data.varpi_cell]
    eta_cell = data.eta_cell.tolist()
    o_bar = data.o_bar.tolist()
    theta_lookup = {(l, c, p, r, k): data.handling(l, c, r, k) for (l, c, p, r, k) in h_index}
    err_lookup = {(l, c, p, r, k): 1.0 - data.accuracy(l, c, r, k) for (l, c, p, r, k) in h_index}
    audit_lookup = {(l, c, p, r, k): data.audit_rate(c, r) for (l, c, p, r, k) in h_index}

    for d in range(nD):
        for w in range(nW):
            g = int(data.gidx[d][w])
            volume = (data.psi * (float(data.Z[d][w]) * data.a_bar[:, d])[:, None]).tolist()
            esc = data.phi_esc[:, :, :, g].tolist()
            mid = data.phi_mid[:, :, :, g].tolist()
            blk = data.phi_blk[:, :, :, g].tolist()
            pas = data.phi_pas[:, :, :, g].tolist()
            efn = data.e_fn[:, :, :, g].tolist()
            efp = data.e_fp[:, :, :, g].tolist()
            pih = data.pi_hat[:, :, :, g].tolist()

            lam = model.addVars(lcp_all, lb=0.0, ub=1.0,
                                vtype=(GRB.CONTINUOUS if LAM_CONTINUOUS else GRB.BINARY))
            if lock_pref:
                for (l, c, p) in lcp_all:
                    if p == data.p_ref:
                        lam[l, c, p].LB = 1.0
                    else:
                        lam[l, c, p].UB = 0.0
            h = model.addVars(h_index, lb=0.0)
            uplus = model.addVars(up_index, lb=0.0)
            uminus = model.addVars(um_index, lb=0.0)
            over = model.addVars(nL, nK, lb=0.0)
            insp = model.addVars(nL, lb=0.0)
            scell = model.addVars(cell_index, lb=0.0)
            slng = model.addVars(nL, lb=0.0)
            stilde = model.addVars(nL, lb=0.0)
            sqc = model.addVars(nL, lb=0.0)
            total_adj = model.addVar(lb=0.0)
            if keep_blocks:
                blocks[(d, w)] = {"lam": lam, "h": h, "uplus": uplus, "uminus": uminus,
                                  "over": over, "insp": insp, "scell": scell,
                                  "slng": slng, "stilde": stilde, "sqc": sqc}

            coefs = []
            terms = []
            for (l, c, p) in lcp_all:
                base = volume[l][c]
                coefs.append(data.c_ml * base * esc[l][c][p]
                             + base * (c_fn[c] * efn[l][c][p] + c_fp[c] * efp[l][c][p]))
                terms.append(lam[l, c, p])
            for l in range(nL):
                for k in range(nK):
                    coefs.append(float(c_ot[l][k]))
                    terms.append(over[l, k])
            for key in h_index:
                l, c, p, r, k = key
                value = err_lookup[key] * (pih[l][c][p] * c_fn[c]
                                           + (1.0 - pih[l][c][p]) * c_fp[c])
                if r == "out":
                    value += c_out[l] * theta_lookup[key]
                coefs.append(value)
                terms.append(h[key])
            for (l, c, p) in up_index:
                coefs.append(c_fn[c] * pih[l][c][p])
                terms.append(uplus[l, c, p])
            for (l, c, p) in um_index:
                coefs.append(c_fp[c] * (1.0 - pih[l][c][p]))
                terms.append(uminus[l, c, p])
            for l in range(nL):
                coefs.append(data.varpi_lng)
                terms.append(stilde[l])
                coefs.append(data.varpi_qc)
                terms.append(sqc[l])
            for (l, c) in cell_index:
                coefs.append(varpi_cell[c])
                terms.append(scell[l, c])
            model.addConstr(jcost[d, w] == gp.LinExpr(coefs, terms))

            for l in range(nL):
                for c in range(nC):
                    model.addConstr(gp.quicksum(lam[l, c, p] for p in range(nP)) == 1.0)

            for (l, c, p) in lcp_all:
                flow = gp.quicksum(h[l, c, p, r, k] for (r, k) in data.A_c[c])
                if c not in data.C_fc:
                    flow += uplus[l, c, p]
                if c not in data.C_fo:
                    flow += uminus[l, c, p]
                model.addConstr(flow == volume[l][c] * mid[l][c][p] * lam[l, c, p])

            for l in range(nL):
                for c in data.C_free:
                    model.addConstr(
                        gp.quicksum(uplus[l, c, p] for p in range(nP))
                        <= float(data.U_bar[l][c][d]) * (1.0 - y[l, c]))
                    model.addConstr(
                        gp.quicksum(uminus[l, c, p] for p in range(nP))
                        <= float(data.U_bar[l][c][d]) * y[l, c])

            for l in range(nL):
                for k in range(nK):
                    load = gp.quicksum(
                        float(data.theta_in[l][c][k]) * h[l, c, p, "in", k + 1]
                        for c in data.C_k[k + 1] for p in range(nP))
                    if k == nK - 1:
                        load += insp[l]
                    model.addConstr(load <= data.H * n[l, k] + over[l, k])
                    model.addConstr(over[l, k] <= float(o_bar[l][k]) * n[l, k])

            for l in range(nL):
                acoefs = []
                aterms = []
                for c in range(nC):
                    for p in range(nP):
                        acoefs.append(theta_qc[l][c] * (
                            vs_blk[c] * volume[l][c] * blk[l][c][p]
                            + vs_pas[c] * volume[l][c] * pas[l][c][p]))
                        aterms.append(lam[l, c, p])
                        if c not in data.C_fc:
                            acoefs.append(theta_qc[l][c] * vs_pas[c])
                            aterms.append(uplus[l, c, p])
                        if c not in data.C_fo:
                            acoefs.append(theta_qc[l][c] * vs_blk[c])
                            aterms.append(uminus[l, c, p])
                        for (r, k) in data.A_c[c]:
                            acoefs.append(theta_qc[l][c] * audit_lookup[(l, c, p, r, k)])
                            aterms.append(h[l, c, p, r, k])
                model.addConstr(insp[l] + sqc[l] >= gp.LinExpr(acoefs, aterms))

            for l in range(nL):
                model.addConstr(
                    gp.quicksum(float(data.theta_out[l][c]) * h[l, c, p, "out", 1]
                                for (c, p) in out_by_pool[l])
                    <= float(data.v_bar[l][d][w]))

            qbar = [[volume[l][c] * mid[l][c][data.p_ref] for c in range(nC)] for l in range(nL)]
            pool = [sum(qbar[l]) for l in range(nL)]
            grand = sum(pool)

            for (l, c) in cell_index:
                model.addConstr(
                    gp.quicksum(h[l, c, p, r, k] for p in range(nP) for (r, k) in data.A_c[c])
                    + scell[l, c] >= float(eta_cell[l][c]) * qbar[l][c])

            model.addConstr(total_adj == gp.quicksum(h[key] for key in h_index))

            for l in range(nL):
                pool_adj = gp.quicksum(h[l, c, p, r, k] for c in range(nC)
                                       for p in range(nP) for (r, k) in data.A_c[c])
                share = (pool[l] / grand) if grand > 0.0 else 0.0
                model.addConstr(pool_adj + slng[l]
                                >= (1.0 - data.epsilon_lng) * share * total_adj)
                model.addConstr(stilde[l] >= slng[l]
                                - gp.quicksum(scell[l, c] for c in data.C_res))

    model.update()
    model._first_stage = {"n": n, "y": y, "xi": xi, "wtail": wtail, "jcost": jcost}
    model._blocks = blocks
    return model


def apply_parameters(model, params):
    model.Params.TimeLimit = float(params.get("time_limit", GRB.INFINITY))
    model.Params.MIPGap = float(params.get("mip_gap", 1e-4))
    model.Params.Threads = int(params.get("threads", 0))
    model.Params.OutputFlag = 1 if params.get("output", False) else 0
    node_start = params.get("nodefile_start_gb", None)
    if node_start is not None:
        model.Params.NodefileStart = float(node_start)
    mem_limit = params.get("soft_memory_limit_gb", None)
    if mem_limit is not None:
        model.Params.SoftMemLimit = float(mem_limit)


def solve_instance(instance, params=None):
    global LAM_CONTINUOUS
    if params is None:
        params = {}
    LAM_CONTINUOUS = bool(params.get("lam_continuous", LAM_CONTINUOUS))
    started = time.time()
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if params.get("output", False) else 0)
    env.start()
    model = None
    try:
        data = ModelData(instance)
        model = build_model(data, env, keep_blocks=bool(params.get("keep_blocks", False)))
        build_time = time.time() - started
        apply_parameters(model, params)
        model.optimize()
        solve_time = float(model.Runtime)
        has_solution = model.SolCount > 0
        objective = float(model.ObjVal) if has_solution else None
        try:
            bound = float(model.ObjBound)
        except gp.GurobiError:
            bound = None
        gap = float(model.MIPGap) * 100.0 if has_solution else None
        if gap is not None and (gap != gap or gap > 1e8):
            gap = None
        status = int(model.Status)
        num_vars = int(model.NumVars)
        num_constrs = int(model.NumConstrs)
        num_bin = int(model.NumBinVars) + int(model.NumIntVars)
    finally:
        if model is not None:
            model.dispose()
        env.dispose()
    return {
        "objective": objective,
        "bound": bound,
        "gap": gap,
        "time": time.time() - started,
        "build_time": build_time,
        "solve_time": solve_time,
        "status": status,
        "num_vars": num_vars,
        "num_constrs": num_constrs,
        "num_discrete": num_bin,
    }