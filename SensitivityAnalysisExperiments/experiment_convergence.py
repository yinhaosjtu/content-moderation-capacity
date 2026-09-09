import copy
import json
import math
import os
import time

import numpy as np
from scipy.special import erf

from Solvers import solver_benders as sb

INSTANCE_DIR = "instances"
SCALES = ("Small", "Medium")
INSTANCES_PER_SCALE = 1

N_GRID = [100, 200, 300, 400, 500]
SEEDS = [11, 12, 13, 14, 15]

RESULT_FILE = "results_convergence.xlsx"
CACHE_FILE = "cache/cache_convergence.jsonl"

MIP_GAP = 0.0
THREADS = 0
TIME_LIMIT = 3600.0


def cache_identity(unit, instance):
    return str(unit), str(instance)


def load_cache(path):
    cache = {}
    if not os.path.exists(path):
        return cache
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (TypeError, ValueError):
                continue
            unit = entry.get("unit")
            instance = entry.get("instance")
            if isinstance(unit, str) and isinstance(instance, str) \
                    and "result" in entry:
                cache[cache_identity(unit, instance)] = entry["result"]
    return cache


def _cache_json_default(value):
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(repr(value))


def store_cache(path, unit, instance, result):
    cache_dir = os.path.dirname(path)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    payload = {"unit": str(unit), "instance": str(instance), "result": result}
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb+") as boundary:
            boundary.seek(-1, os.SEEK_END)
            if boundary.read(1) != b"\n":
                boundary.seek(0, os.SEEK_END)
                boundary.write(b"\n")
                boundary.flush()
                os.fsync(boundary.fileno())
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=True,
                                default=_cache_json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def is_cacheable_result(result):
    return isinstance(result, dict) and result.get("_status", "ok") != "error"


def exception_status(error):
    text = "{0}: {1}".format(type(error).__name__, error).lower()
    oom_markers = ("out of memory", "out-of-memory", "memory limit",
                   "mem_limit", "cannot allocate memory", "failed to allocate",
                   "allocation failed", "gurobi error 10001")
    if (isinstance(error, MemoryError)
            or getattr(error, "errno", None) == 10001
            or any(marker in text for marker in oom_markers)):
        return "oom"
    if isinstance(error, TimeoutError) or any(
            m in text for m in ("time limit", "time-limit", "timed out",
                                "timeout")):
        return "time_limit"
    return "error"


def error_text(error):
    return "{0}: {1}".format(type(error).__name__, error)


_CACHE_INTEGER_STATUSES = {
    2: "ok", 3: "infeasible", 4: "inf_or_unbd", 5: "unbounded", 6: "cutoff",
    7: "iteration_limit", 8: "node_limit", 9: "time_limit",
    10: "solution_limit", 11: "interrupted", 12: "numeric", 13: "suboptimal",
    15: "user_obj_limit", 16: "work_limit", 17: "oom",
}


def normalized_solver_status(result, time_limit=None, time_field="time"):
    if not isinstance(result, dict):
        return "error"
    raw = result.get("status")
    status = None
    if isinstance(raw, bool):
        status = None
    elif isinstance(raw, (int, float)) and math.isfinite(float(raw)):
        status = _CACHE_INTEGER_STATUSES.get(
            int(raw), "status_{0}".format(int(raw)))
    elif raw is not None:
        status = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {"optimal": "ok", "loaded": "ok", "timelimit": "time_limit",
                   "memory_limit": "oom", "out_of_memory": "oom",
                   "mem_limit": "oom"}
        status = aliases.get(status, status)
    elapsed = result.get(time_field)
    try:
        elapsed = float(elapsed)
    except (TypeError, ValueError):
        elapsed = None
    if status is None and time_limit is not None and elapsed is not None:
        threshold = max(
            float(time_limit) - max(1.0, 0.001 * float(time_limit)), 0.0)
        if elapsed >= threshold:
            status = "time_limit"
    if status is None:
        status = "ok" if result.get("objective") is not None else "no_solution"
    return status


def find_instances(directory, scale, count):
    found = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        stem = os.path.splitext(name)[0]
        parts = stem.rsplit("_", 1)
        if parts[0] != scale or not parts[1].isdigit():
            continue
        found.append((int(parts[1]), os.path.join(directory, name)))
    found.sort(key=lambda item: item[0])
    if len(found) < count:
        raise FileNotFoundError(
            "found {0} {1} instances but {2} required".format(
                len(found), scale, count))
    return found[:count]


def std_normal_cdf(x):
    return 0.5 * (1.0 + erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def build_scenarios(rng, n_days, n_scen, sigma_z, kappa_delta, beta_delta,
                    rho_zd, varrho, delta_grid, grid_levels):
    cov = np.array([[1.0, rho_zd], [rho_zd, 1.0]])
    chol = np.linalg.cholesky(cov)
    state = rng.standard_normal((n_scen, 2)) @ chol.T
    scores = np.zeros((n_days, n_scen, 2))
    for d in range(n_days):
        innov = rng.standard_normal((n_scen, 2)) @ chol.T
        state = varrho * state + math.sqrt(1.0 - varrho * varrho) * innov
        scores[d] = state
    z_val = np.exp(-0.5 * sigma_z * sigma_z + sigma_z * scores[:, :, 0])
    u_delta = std_normal_cdf(scores[:, :, 1])
    u_delta = np.clip(u_delta, 1e-12, 1.0 - 1e-12)
    pos = np.searchsorted(grid_levels, u_delta)
    pos = np.clip(pos, 1, len(grid_levels) - 1)
    left = grid_levels[pos - 1]
    right = grid_levels[pos]
    idx = np.where(u_delta - left <= right - u_delta, pos - 1, pos)
    return z_val, idx.astype(int), delta_grid[idx]


def resample_scenarios(instance, n_scen, seed):
    # Generate new scenarios for convergence analysis using AR(1) process
    working = copy.deepcopy(instance)
    n_days = int(working["dimensions"]["n_days"])
    unc = working["uncertainty"]
    sigma_z = float(unc["sigma_Z"])
    kappa_delta = float(unc["kappa_delta"])
    beta_delta = float(unc["beta_delta"])
    rho_zd = float(unc["rho_z_delta"])
    varrho = float(unc["varrho"])
    zeta1 = float(unc["zeta1"])
    zeta2 = float(unc["zeta2"])

    delta_grid = np.asarray(working["delta_grid"], dtype=float)
    grid_levels = np.asarray(working["delta_grid_levels"], dtype=float)
    v0_bar = np.asarray(working["workload"]["v0_bar"], dtype=float)

    rng = np.random.default_rng(int(seed))
    z_val, delta_idx, delta_val = build_scenarios(
        rng, n_days, n_scen, sigma_z, kappa_delta, beta_delta,
        rho_zd, varrho, delta_grid, grid_levels)

    v_bar = v0_bar[:, :, None] / (
        1.0 + zeta1 * np.maximum(z_val - 1.0, 0.0)[None, :, :]
        + zeta2 * delta_val[None, :, :])

    working["scenarios"] = {
        "prob": [1.0 / n_scen for _ in range(n_scen)],
        "Z": np.round(z_val, 8).tolist(),
        "delta": np.round(delta_val, 8).tolist(),
        "delta_index": [[int(v) for v in row] for row in delta_idx],
        "v_bar": np.round(v_bar, 4).tolist(),
    }
    working["dimensions"]["n_scenarios"] = int(n_scen)
    return working


def solve_base(instance):
    res = sb.solve_instance(instance, {
        "mip_gap": MIP_GAP,
        "threads": THREADS,
        "time_limit": TIME_LIMIT,
        "extract_channels": True,
    })
    if res.get("objective") is None or res.get("headcount") is None:
        return {"objective": None, "_status": res.get("status", "no_solution")}
    head = res["headcount"]
    dispo = res.get("disposition")
    ch = res.get("channels") or {}
    n_val = {(l, k): int(head[l][k])
             for l in range(len(head)) for k in range(len(head[l]))}
    y_val = {(l, c): int(dispo[l][c])
             for l in range(len(dispo)) for c in range(len(dispo[l]))}
    return {
        "objective": float(res["objective"]),
        "n": n_val,
        "y": y_val,
        "F_commit": ch.get("F_commit"),
        "E_recourse": ch.get("E_recourse"),
        "cvar95": ch.get("cvar95"),
        "_status": "ok",
    }


def run_unit(instance, n_scen, seed):
    working = resample_scenarios(instance, n_scen, seed)
    out = solve_base(working)
    if out.get("objective") is None:
        return {"_status": out.get("_status", "no_solution")}
    n_items = sorted(out["n"].items())
    y_items = sorted(out["y"].items())
    return {
        "objective": float(out["objective"]),
        "n_flat": [int(v) for _, v in n_items],
        "n_total": int(sum(v for _, v in n_items)),
        "y_flat": [int(v) for _, v in y_items],
        "cvar95": out.get("cvar95"),
        "F_commit": out.get("F_commit"),
        "E_recourse": out.get("E_recourse"),
        "_status": out.get("_status", "ok"),
    }


def main():
    from openpyxl import Workbook

    cache = load_cache(CACHE_FILE)
    rows = []

    for scale in SCALES:
        selections = find_instances(INSTANCE_DIR, scale, INSTANCES_PER_SCALE)
        for _seed_id, path in selections:
            with open(path, "r", encoding="utf-8") as handle:
                stored = json.load(handle)
            name = stored["meta"]["name"]
            for n_scen in N_GRID:
                for rep_idx, seed in enumerate(SEEDS, start=1):
                    unit = "conv"
                    inst_key = "{0}|N={1}|rep={2}".format(name, n_scen, rep_idx)
                    identity = cache_identity(unit, inst_key)
                    if identity in cache:
                        res = cache[identity]
                        tag = "cache"
                    else:
                        started = time.perf_counter()
                        try:
                            res = run_unit(stored, n_scen, seed)
                        except Exception as error:
                            status = exception_status(error)
                            if status not in ("oom", "time_limit"):
                                raise
                            res = {"_status": status,
                                   "_error": error_text(error)}
                        res["_time_s"] = time.perf_counter() - started
                        if is_cacheable_result(res):
                            store_cache(CACHE_FILE, unit, inst_key, res)
                            cache[identity] = res
                        tag = "solved"
                    obj = res.get("objective")
                    print("[{0}] {1} N={2} rep={3} n_total={4} ({5})".format(
                        scale, name, n_scen, rep_idx,
                        res.get("n_total", "NA"), tag), flush=True)
                    rows.append({
                        "scale": scale, "instance": name, "N": n_scen,
                        "rep": rep_idx, "n_total": res.get("n_total"),
                    })

    book = Workbook()
    sh = book.active
    sh.title = "convergence"
    sh.append(["scale", "instance", "N", "rep", "n_total"])
    for r in rows:
        sh.append([r["scale"], r["instance"], r["N"], r["rep"],
                   r["n_total"]])
    book.save(RESULT_FILE)
    print("saved", RESULT_FILE)


if __name__ == "__main__":
    main()
