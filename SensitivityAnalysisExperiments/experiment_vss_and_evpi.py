import copy
import json
import math
import os
import re
import tempfile
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from Solvers import solver_gurobi


_CACHE_INTEGER_STATUSES = {
    2: "ok",
    3: "infeasible",
    4: "inf_or_unbd",
    5: "unbounded",
    6: "cutoff",
    7: "iteration_limit",
    8: "node_limit",
    9: "time_limit",
    10: "solution_limit",
    11: "interrupted",
    12: "numeric",
    13: "suboptimal",
    15: "user_obj_limit",
    16: "work_limit",
    17: "oom",
}


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
            if isinstance(unit, str) and isinstance(instance, str) and "result" in entry:
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
        handle.write(json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=True,
            default=_cache_json_default,
        ) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def exception_status(error):
    text = "{0}: {1}".format(type(error).__name__, error).lower()
    oom_markers = (
        "out of memory", "out-of-memory", "memory limit", "mem_limit",
        "cannot allocate memory", "failed to allocate", "allocation failed",
        "gurobi error 10001",
    )
    if (isinstance(error, MemoryError)
            or getattr(error, "errno", None) == 10001
            or any(marker in text for marker in oom_markers)):
        return "oom"
    timeout_markers = ("time limit", "time-limit", "timed out", "timeout")
    if isinstance(error, TimeoutError) or any(
            marker in text for marker in timeout_markers):
        return "time_limit"
    return "error"


def error_text(error):
    return "{0}: {1}".format(type(error).__name__, error)


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
        aliases = {
            "optimal": "ok",
            "loaded": "ok",
            "timelimit": "time_limit",
            "memory_limit": "oom",
            "out_of_memory": "oom",
            "mem_limit": "oom",
        }
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


def is_cacheable_result(result):
    return isinstance(result, dict) and result.get("_status", "ok") != "error"

INSTANCE_DIR_CANDIDATES = ("instances", "instance")
INSTANCE_SCALES = ("small", "medium")
INSTANCES_PER_SCALE = 5
RESULT_FILE = "results_vss_and_evpi.xlsx"
RESULT_SHEET = "results"
RESULT_COLUMNS = ("Instance", "RP", "EV", "EEV", "WS", "VSS(%)", "EVPI(%)")
RESULT_COLUMN_WIDTHS = (16, 22, 22, 22, 22, 14, 14)
CACHE_FILE = "cache/cache_vss_and_evpi.jsonl"


MIP_GAP = 0
THREADS = 0
SOLVER_OUTPUT = False
SOFT_MEMORY_LIMIT_GB = None
NODEFILE_START_GB = None

PROBABILITY_TOLERANCE = 1e-8
ABSOLUTE_TOLERANCE = 1e-6
RELATIVE_TOLERANCE = 1e-6
INTEGRALITY_TOLERANCE = 1e-4
FIXING_TOLERANCE = 1e-6

DETECTION_KEYS = ("phi_esc", "phi_mid", "phi_blk", "phi_pas", "e_fn", "e_fp", "pi_hat")


def build_status_names():
    names = {}
    for key in ("LOADED", "OPTIMAL", "INFEASIBLE", "INF_OR_UNBD", "UNBOUNDED",
                "CUTOFF", "ITERATION_LIMIT", "NODE_LIMIT", "TIME_LIMIT",
                "SOLUTION_LIMIT", "INTERRUPTED", "NUMERIC", "SUBOPTIMAL",
                "INPROGRESS", "USER_OBJ_LIMIT", "WORK_LIMIT", "MEM_LIMIT"):
        code = getattr(GRB, key, None)
        if code is not None:
            names[int(code)] = key
    return names


STATUS_NAMES = build_status_names()


def is_valid_value(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def format_metric(value, digits):
    if not is_valid_value(value):
        return "nan"
    return "{0:.{1}f}".format(value, digits)


def parse_instance_file_name(file_name):
    stem = os.path.splitext(file_name)[0]
    match = re.match(r"^(.*)_(\d+)$", stem)
    if match is None:
        return stem, None
    return match.group(1), int(match.group(2))


def select_instances():
    searched = []
    grouped = {scale: [] for scale in INSTANCE_SCALES}
    for directory in INSTANCE_DIR_CANDIDATES:
        if not os.path.isdir(directory):
            continue
        searched.append(directory)
        for file_name in sorted(os.listdir(directory)):
            if not file_name.endswith(".json"):
                continue
            scale, seed = parse_instance_file_name(file_name)
            if seed is None:
                continue
            key = scale.lower()
            if key in grouped:
                grouped[key].append((seed, file_name, os.path.join(directory, file_name)))
    if not searched:
        raise FileNotFoundError(
            "No instance directory was found among the candidates: {0}".format(
                ", ".join(INSTANCE_DIR_CANDIDATES)))
    selections = []
    for scale in INSTANCE_SCALES:
        candidates = grouped[scale]
        if len(candidates) < INSTANCES_PER_SCALE:
            raise FileNotFoundError(
                "Found {0} {1}-scale instance files with a numeric seed suffix in {2} "
                "but {3} are required".format(
                    len(candidates), scale, ", ".join(searched), INSTANCES_PER_SCALE))
        candidates.sort(key=lambda item: (item[0], item[1]))
        for ordinal, (_seed, file_name, path) in enumerate(
                candidates[:INSTANCES_PER_SCALE], start=1):
            prefix, _suffix = parse_instance_file_name(file_name)
            label = "{0}_{1:02d}".format(prefix, ordinal)
            selections.append((label, path))
    return selections


def load_instance(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_probabilities(instance):
    raw_values = instance["scenarios"]["prob"]
    values = [float(v) for v in raw_values]
    n_scenarios = int(instance["dimensions"]["n_scenarios"])
    if len(values) != n_scenarios:
        raise ValueError(
            "Scenario probability vector has length {0} but n_scenarios is {1}".format(
                len(values), n_scenarios))
    if not all(math.isfinite(v) and v >= 0.0 for v in values):
        raise ValueError("Scenario probabilities contain negative or non-finite entries")
    total = math.fsum(values)
    deviation = abs(total - 1.0)
    if deviation > PROBABILITY_TOLERANCE:
        raise ValueError(
            "Scenario probabilities sum to {0!r}, deviating from one beyond tolerance {1}".format(
                total, PROBABILITY_TOLERANCE))
    normalization_applied = False
    normalized = values
    if deviation > 0.0:
        normalized = [v / total for v in values]
        normalization_applied = True
    return normalized, normalization_applied


def validate_random_parameter_dimensions(instance):
    dims = instance["dimensions"]
    n_languages = int(dims["n_languages"])
    n_days = int(dims["n_days"])
    n_scenarios = int(dims["n_scenarios"])
    scenarios = instance["scenarios"]
    z_values = np.asarray(scenarios["Z"], dtype=float)
    delta_index = np.asarray(scenarios["delta_index"], dtype=int)
    v_values = np.asarray(scenarios["v_bar"], dtype=float)
    issues = []
    if z_values.shape != (n_days, n_scenarios):
        issues.append("Z has shape {0} but ({1}, {2}) is required".format(
            z_values.shape, n_days, n_scenarios))
    if delta_index.shape != (n_days, n_scenarios):
        issues.append("delta_index has shape {0} but ({1}, {2}) is required".format(
            delta_index.shape, n_days, n_scenarios))
    if v_values.shape != (n_languages, n_days, n_scenarios):
        issues.append("v_bar has shape {0} but ({1}, {2}, {3}) is required".format(
            v_values.shape, n_languages, n_days, n_scenarios))
    grid_length = None
    for key in DETECTION_KEYS:
        array = np.asarray(instance["detection_coefficients"][key], dtype=float)
        if array.ndim != 4:
            issues.append("{0} has {1} axes but 4 are required".format(key, array.ndim))
            continue
        if grid_length is None:
            grid_length = int(array.shape[3])
        elif int(array.shape[3]) != grid_length:
            issues.append("{0} last-axis length {1} differs from {2}".format(
                key, int(array.shape[3]), grid_length))
    if grid_length is not None and delta_index.size > 0:
        if int(delta_index.min()) < 0 or int(delta_index.max()) >= grid_length:
            issues.append("delta_index values fall outside the valid range [0, {0})".format(
                grid_length))
    if not np.all(np.isfinite(z_values)) or not np.all(np.isfinite(v_values)):
        issues.append("Z or v_bar contains non-finite entries")
    if issues:
        raise ValueError(
            "Random parameter dimension validation failed: {0}".format("; ".join(issues)))


def make_recourse_instance(original, probabilities):
    working = copy.deepcopy(original)
    working["scenarios"]["prob"] = [float(p) for p in probabilities]
    return working


def make_expected_value_instance(original, probabilities):
    working = copy.deepcopy(original)
    n_days = int(working["dimensions"]["n_days"])
    prob = np.asarray(probabilities, dtype=float)
    z_values = np.asarray(working["scenarios"]["Z"], dtype=float)
    delta_index = np.asarray(working["scenarios"]["delta_index"], dtype=int)
    v_values = np.asarray(working["scenarios"]["v_bar"], dtype=float)
    z_mean = z_values @ prob
    v_mean = v_values @ prob
    for key in DETECTION_KEYS:
        array = np.asarray(working["detection_coefficients"][key], dtype=float)
        selected = array[:, :, :, delta_index]
        expected = np.tensordot(selected, prob, axes=([4], [0]))
        working["detection_coefficients"][key] = expected.tolist()
    if "delta" in working["scenarios"]:
        delta_values = np.asarray(working["scenarios"]["delta"], dtype=float)
        working["scenarios"]["delta"] = (delta_values @ prob)[:, None].tolist()
    working["scenarios"]["prob"] = [1.0]
    working["scenarios"]["Z"] = z_mean[:, None].tolist()
    working["scenarios"]["delta_index"] = [[d] for d in range(n_days)]
    working["scenarios"]["v_bar"] = v_mean[:, :, None].tolist()
    working["dimensions"]["n_scenarios"] = 1
    return working


def make_single_scenario_instance(original, scenario_index):
    working = copy.deepcopy(original)
    scenarios = working["scenarios"]
    n_languages = int(working["dimensions"]["n_languages"])
    n_days = int(working["dimensions"]["n_days"])
    scenarios["prob"] = [1.0]
    scenarios["Z"] = [[float(scenarios["Z"][d][scenario_index])] for d in range(n_days)]
    scenarios["delta_index"] = [[int(scenarios["delta_index"][d][scenario_index])]
                                for d in range(n_days)]
    if "delta" in scenarios:
        scenarios["delta"] = [[float(scenarios["delta"][d][scenario_index])]
                              for d in range(n_days)]
    scenarios["v_bar"] = [[[float(scenarios["v_bar"][l][d][scenario_index])]
                           for d in range(n_days)] for l in range(n_languages)]
    working["dimensions"]["n_scenarios"] = 1
    return working


def configure_solver_parameters(model):
    model.Params.MIPGap = float(MIP_GAP)
    model.Params.Threads = int(THREADS)
    model.Params.OutputFlag = 1 if SOLVER_OUTPUT else 0
    if NODEFILE_START_GB is not None:
        model.Params.NodefileStart = float(NODEFILE_START_GB)
    if SOFT_MEMORY_LIMIT_GB is not None:
        model.Params.SoftMemLimit = float(SOFT_MEMORY_LIMIT_GB)


def solve_rp_bbc(instance_data, label):
    from Solvers import solver_benders as sb
    result = {
        "label": label, "status_code": None, "status_name": "BBC",
        "optimal": False, "model_sense": int(GRB.MINIMIZE),
        "objective": float("nan"), "first_stage_cost": float("nan"),
        "second_stage_component": float("nan"), "expected_recourse": float("nan"),
        "tail_component": float("nan"), "first_stage_n": None,
        "first_stage_y": None, "max_fixed_deviation": None,
    }
    res = sb.solve_instance(instance_data, {
        "mip_gap": 1e-4, "threads": THREADS, "time_limit": 3600.0,
        "extract_channels": True})
    result["terminal_status"] = normalized_solver_status(
        res, 3600.0, "time")
    ch = res.get("channels")
    if res.get("objective") is None or ch is None:
        return result
    gamma = float(instance_data["uncertainty"]["gamma"])
    head, dispo = res["headcount"], res["disposition"]
    nK = len(head[0]) if head else 0
    result.update({
        "optimal": True,
        "objective": res["objective"],
        "first_stage_cost": ch.get("F_commit"),
        "expected_recourse": ch.get("E_recourse"),
        "second_stage_component": (1.0 - gamma) * ch.get("E_recourse", 0.0)
        + gamma * ch.get("cvar95", 0.0),
        "tail_component": ch.get("cvar95"),
        "first_stage_n": {(l, k): float(head[l][k])
                          for l in range(len(head)) for k in range(nK)},
        "first_stage_y": {(l, c): float(dispo[l][c])
                          for l in range(len(dispo)) for c in range(len(dispo[l]))},
    })
    return result


def evaluate_eev_bbc(instance_data, fixed_first_stage, label):
    from Solvers import solver_benders as sb
    nL = int(instance_data["dimensions"]["n_languages"])
    nK = int(instance_data["dimensions"]["n_tiers"])
    nC = int(instance_data["dimensions"]["n_categories"])
    n_nested = [[int(round(fixed_first_stage["n"][(l, k)])) for k in range(nK)]
                for l in range(nL)]
    y_nested = [[int(round(fixed_first_stage["y"][(l, c)])) for c in range(nC)]
                for l in range(nL)]
    res = sb.solve_instance(instance_data, {
        "threads": THREADS, "eval_only": {"n": n_nested, "y": y_nested}})
    result = {
        "label": label, "status_code": None, "status_name": "BBC_EVAL",
        "optimal": False, "model_sense": int(GRB.MINIMIZE),
        "objective": float("nan"), "first_stage_cost": float("nan"),
        "second_stage_component": float("nan"), "expected_recourse": float("nan"),
        "tail_component": float("nan"), "first_stage_n": None,
        "first_stage_y": None, "max_fixed_deviation": 0.0,
        "terminal_status": normalized_solver_status(res),
    }
    ch = res.get("channels")
    if res.get("objective") is None or ch is None:
        return result
    gamma = float(instance_data["uncertainty"]["gamma"])
    result.update({
        "optimal": True,
        "objective": res["objective"],
        "first_stage_cost": ch.get("F_commit"),
        "expected_recourse": ch.get("E_recourse"),
        "second_stage_component": (1.0 - gamma) * ch.get("E_recourse", 0.0)
        + gamma * ch.get("cvar95", 0.0),
        "tail_component": ch.get("cvar95"),
    })
    return result


def solve_two_stage_model(instance_data, label, fixed_first_stage=None):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if SOLVER_OUTPUT else 0)
    env.start()
    model = None
    result = {
        "label": label,
        "status_code": None,
        "status_name": "NOT_SOLVED",
        "optimal": False,
        "model_sense": None,
        "objective": float("nan"),
        "first_stage_cost": float("nan"),
        "second_stage_component": float("nan"),
        "expected_recourse": float("nan"),
        "tail_component": float("nan"),
        "first_stage_n": None,
        "first_stage_y": None,
        "max_fixed_deviation": None,
    }
    try:
        data = solver_gurobi.ModelData(instance_data)
        model = solver_gurobi.build_model(data, env)
        result["model_sense"] = int(model.ModelSense)
        if fixed_first_stage is not None:
            n_vars = model._first_stage["n"]
            y_vars = model._first_stage["y"]
            for key in sorted(n_vars.keys()):
                value = float(fixed_first_stage["n"][key])
                n_vars[key].LB = value
                n_vars[key].UB = value
            for key in sorted(y_vars.keys()):
                value = float(fixed_first_stage["y"][key])
                y_vars[key].LB = value
                y_vars[key].UB = value
            model.update()
        configure_solver_parameters(model)
        model.optimize()
        status = int(model.Status)
        result["status_code"] = status
        result["status_name"] = STATUS_NAMES.get(status, "STATUS_{0}".format(status))
        result["optimal"] = status == GRB.OPTIMAL and int(model.SolCount) > 0
        if result["optimal"]:
            first_stage = model._first_stage
            n_solution = {key: float(first_stage["n"][key].X)
                          for key in sorted(first_stage["n"].keys())}
            y_solution = {key: float(first_stage["y"][key].X)
                          for key in sorted(first_stage["y"].keys())}
            xi_value = float(first_stage["xi"].X)
            wtail_values = [float(first_stage["wtail"][w].X) for w in range(data.nW)]
            scenario_recourse = [
                math.fsum(float(first_stage["jcost"][d, w].X) for d in range(data.nD))
                for w in range(data.nW)]
            probabilities = [float(p) for p in data.prob]
            commitment = math.fsum(
                float(data.c_hr[l][k]) * n_solution[(l, k)]
                for l in range(data.nL) for k in range(data.nK))
            expected_recourse = math.fsum(
                probabilities[w] * scenario_recourse[w] for w in range(data.nW))
            tail_component = xi_value + (1.0 / (1.0 - data.alpha)) * math.fsum(
                probabilities[w] * wtail_values[w] for w in range(data.nW))
            second_stage = ((1.0 - data.gamma) * expected_recourse
                            + data.gamma * tail_component)
            result["objective"] = float(model.ObjVal)
            result["first_stage_cost"] = commitment
            result["expected_recourse"] = expected_recourse
            result["tail_component"] = tail_component
            result["second_stage_component"] = second_stage
            result["first_stage_n"] = n_solution
            result["first_stage_y"] = y_solution
            if fixed_first_stage is not None:
                deviations = [abs(n_solution[key] - float(fixed_first_stage["n"][key]))
                              for key in n_solution]
                deviations.extend(abs(y_solution[key] - float(fixed_first_stage["y"][key]))
                                  for key in y_solution)
                result["max_fixed_deviation"] = max(deviations) if deviations else 0.0
    finally:
        if model is not None:
            model.dispose()
        env.dispose()
    return result


def check_objective_decomposition(name, result):
    if not result["optimal"]:
        return
    objective = result["objective"]
    recomposed = result["first_stage_cost"] + result["second_stage_component"]
    difference = abs(objective - recomposed)
    tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * max(1.0, abs(objective))
    if difference > tolerance:
        raise RuntimeError(
            "{0} failed because objective {1!r} differs from the decomposition {2!r} "
            "beyond tolerance {3!r}".format(name, objective, recomposed, tolerance))


def extract_fixed_first_stage(result, label):
    fixed = {"n": {}, "y": {}}
    for key in sorted(result["first_stage_n"].keys()):
        value = result["first_stage_n"][key]
        rounded = round(value)
        if abs(value - rounded) > INTEGRALITY_TOLERANCE:
            raise RuntimeError(
                "{0} first-stage variable n{1} has value {2!r} which is not integral "
                "within tolerance {3}".format(label, key, value, INTEGRALITY_TOLERANCE))
        fixed["n"][key] = float(rounded)
    for key in sorted(result["first_stage_y"].keys()):
        value = result["first_stage_y"][key]
        rounded = round(value)
        if rounded not in (0, 1) or abs(value - rounded) > INTEGRALITY_TOLERANCE:
            raise RuntimeError(
                "{0} first-stage variable y{1} has value {2!r} which is not binary "
                "within tolerance {3}".format(label, key, value, INTEGRALITY_TOLERANCE))
        fixed["y"][key] = float(rounded)
    return fixed


def check_eev_fixed_first_stage(eev_result):
    if eev_result is None or not eev_result["optimal"]:
        return
    deviation = eev_result["max_fixed_deviation"]
    if deviation is None or deviation > FIXING_TOLERANCE:
        raise RuntimeError(
            "EEV first-stage variables deviate from the EV solution by {0!r}, exceeding "
            "tolerance {1}".format(deviation, FIXING_TOLERANCE))


def solve_wait_and_see(original, probabilities):
    n_scenarios = int(original["dimensions"]["n_scenarios"])
    records = []
    for scenario_index in range(n_scenarios):
        scenario_instance = make_single_scenario_instance(original, scenario_index)
        label = "WS_scenario_{0}".format(scenario_index)
        result = solve_two_stage_model(scenario_instance, label)
        records.append({
            "scenario": scenario_index,
            "probability": float(probabilities[scenario_index]),
            "status_name": result["status_name"],
            "optimal": result["optimal"],
            "objective": result["objective"],
            "model_sense": result["model_sense"],
        })
    return records


def aggregate_wait_and_see(records):
    failed = [record for record in records
              if record["probability"] > 0.0 and not record["optimal"]]
    if failed:
        return float("nan")
    probabilities = np.asarray([record["probability"] for record in records], dtype=float)
    objectives = np.asarray(
        [record["objective"] if record["probability"] > 0.0 else 0.0 for record in records],
        dtype=float)
    weighted_array = probabilities * objectives
    weighted_list = [record["probability"] * record["objective"]
                     if record["probability"] > 0.0 else 0.0 for record in records]
    max_product_error = max(
        (abs(float(weighted_array[i]) - weighted_list[i]) for i in range(len(records))),
        default=0.0)
    product_tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * max(
        [1.0] + [abs(v) for v in weighted_list])
    ws_value = math.fsum(weighted_list)
    ws_reference = float(np.dot(probabilities, objectives))
    aggregation_error = abs(ws_value - ws_reference)
    aggregation_tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * max(1.0, abs(ws_value))
    if max_product_error > product_tolerance or aggregation_error > aggregation_tolerance:
        raise RuntimeError("Wait-and-see aggregation validation failed")
    return ws_value


def resolve_objective_direction(sense_values):
    senses = {int(value) for value in sense_values if value is not None}
    if not senses:
        raise RuntimeError("Objective direction could not be identified because no model was built")
    if len(senses) > 1:
        raise RuntimeError(
            "Inconsistent objective senses were detected across models: {0}".format(
                sorted(senses)))
    sense = senses.pop()
    if sense == int(GRB.MINIMIZE):
        return "minimization"
    if sense == int(GRB.MAXIMIZE):
        return "maximization"
    raise RuntimeError("Unsupported objective sense value {0}".format(sense))


def compute_value_metrics(direction, rp_value, eev_value, ws_value):
    vss_value = float("nan")
    evpi_value = float("nan")
    if direction == "minimization":
        if is_valid_value(rp_value) and is_valid_value(eev_value):
            vss_value = eev_value - rp_value
        if is_valid_value(rp_value) and is_valid_value(ws_value):
            evpi_value = rp_value - ws_value
    else:
        if is_valid_value(rp_value) and is_valid_value(eev_value):
            vss_value = rp_value - eev_value
        if is_valid_value(rp_value) and is_valid_value(ws_value):
            evpi_value = ws_value - rp_value
    return vss_value, evpi_value


def compute_relative_metrics(rp_value, vss_value, evpi_value):
    vss_percent = float("nan")
    evpi_percent = float("nan")
    if is_valid_value(rp_value) and abs(rp_value) >= ABSOLUTE_TOLERANCE:
        if is_valid_value(vss_value):
            vss_percent = 100.0 * vss_value / abs(rp_value)
        if is_valid_value(evpi_value):
            evpi_percent = 100.0 * evpi_value / abs(rp_value)
    return vss_percent, evpi_percent


def make_risk_neutral_recourse_instance(original, probabilities):
    working = make_recourse_instance(original, probabilities)
    working["uncertainty"]["gamma"] = 0.0
    return working


def run_instance(path):
    original_instance = load_instance(path)
    probabilities, normalization_applied = validate_probabilities(original_instance)
    validate_random_parameter_dimensions(original_instance)
    recourse_instance = make_risk_neutral_recourse_instance(
        original_instance, probabilities)
    rp_result = solve_rp_bbc(recourse_instance, "RP")
    ev_result = solve_two_stage_model(
        make_expected_value_instance(original_instance, probabilities), "EV")
    eev_result = None
    if ev_result["optimal"]:
        fixed_first_stage = extract_fixed_first_stage(ev_result, "EV")
        eev_result = evaluate_eev_bbc(
            recourse_instance, fixed_first_stage, "EEV")
    ws_records = solve_wait_and_see(original_instance, probabilities)
    ws_value = aggregate_wait_and_see(ws_records)
    sense_values = [rp_result["model_sense"], ev_result["model_sense"]]
    if eev_result is not None:
        sense_values.append(eev_result["model_sense"])
    sense_values.extend(record["model_sense"] for record in ws_records)
    direction = resolve_objective_direction(sense_values)
    rp_value = rp_result["objective"] if rp_result["optimal"] else float("nan")
    ev_value = ev_result["objective"] if ev_result["optimal"] else float("nan")
    eev_value = (eev_result["objective"]
                 if eev_result is not None and eev_result["optimal"] else float("nan"))
    vss_value, evpi_value = compute_value_metrics(direction, rp_value, eev_value, ws_value)
    vss_percent, evpi_percent = compute_relative_metrics(rp_value, vss_value, evpi_value)
    statuses = [rp_result.get("terminal_status"), ev_result.get("status_name")]
    if eev_result is not None:
        statuses.append(eev_result.get("terminal_status"))
    statuses.extend(record.get("status_name") for record in ws_records)
    normalized = {str(value).strip().lower() for value in statuses if value is not None}
    if any(value in ("oom", "mem_limit", "memory_limit") for value in normalized):
        terminal_status = "oom"
    elif any(value in ("time_limit", "timelimit") for value in normalized):
        terminal_status = "time_limit"
    elif all(is_valid_value(value) for value in (rp_value, ev_value, eev_value, ws_value)):
        terminal_status = "ok"
    else:
        terminal_status = "no_solution"
    return {
        "rp": rp_value,
        "ev": ev_value,
        "eev": eev_value,
        "ws": ws_value,
        "vss_percent": vss_percent,
        "evpi_percent": evpi_percent,
        "direction": direction,
        "normalization_applied": normalization_applied,
        "_status": terminal_status,
    }


def write_results_workbook(rows, directions, normalization_flags):
    book = Workbook()
    sheet = book.active
    sheet.title = RESULT_SHEET
    sheet.append(list(RESULT_COLUMNS))
    header_font = Font(bold=True)
    header_fill = PatternFill(start_color="FFDCE6F1", end_color="FFDCE6F1", fill_type="solid")
    header_alignment = Alignment(horizontal="center", vertical="center")
    for column_index in range(1, len(RESULT_COLUMNS) + 1):
        cell = sheet.cell(row=1, column=column_index)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_alignment
        sheet.column_dimensions[get_column_letter(column_index)].width = (
            RESULT_COLUMN_WIDTHS[column_index - 1])
    for row in rows:
        sheet.append([row[0]] + [value if is_valid_value(value) else None
                                 for value in row[1:]])
    for row_index in range(2, len(rows) + 2):
        for column_index in range(2, len(RESULT_COLUMNS) + 1):
            cell = sheet.cell(row=row_index, column=column_index)
            if cell.value is not None:
                cell.number_format = "0.000000"
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = "A1:{0}{1}".format(
        get_column_letter(len(RESULT_COLUMNS)), max(len(rows) + 1, 2))
    direction_text = ",".join(sorted(set(directions))) if directions else "unknown"
    normalization_applied = any(normalization_flags)
    book.properties.title = "Value of the stochastic solution and expected value of perfect information"
    book.properties.subject = "objective_direction={0}".format(direction_text)
    book.properties.description = (
        "objective_direction={0}; probability_normalization_applied={1}; "
        "risk_neutral_gamma=0.0".format(
            direction_text, str(normalization_applied).lower()))
    book.properties.category = "two_stage_stochastic_programming_evaluation"
    directory = os.path.dirname(os.path.abspath(RESULT_FILE))
    handle, temporary_path = tempfile.mkstemp(
        prefix="results_vss_and_evpi_", suffix=".xlsx", dir=directory)
    os.close(handle)
    try:
        book.save(temporary_path)
        os.replace(temporary_path, RESULT_FILE)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise
    print("Results written to {0}".format(RESULT_FILE), flush=True)


def main():
    selections = select_instances()
    total = len(selections)
    cache = load_cache(CACHE_FILE)
    rows = []
    directions = []
    normalization_flags = []
    for index, (label, path) in enumerate(selections, start=1):
        stem = os.path.splitext(os.path.basename(path))[0]
        unit = "vss_and_evpi"
        identity = cache_identity(unit, stem)
        if identity in cache:
            outcome = cache[identity]
            tag = "cache"
        else:
            started = time.perf_counter()
            try:
                outcome = run_instance(path)
            except Exception as error:
                outcome = {
                    "_status": exception_status(error),
                    "_time_s": time.perf_counter() - started,
                    "_error": error_text(error),
                }
            if is_cacheable_result(outcome):
                store_cache(CACHE_FILE, unit, stem, outcome)
                cache[identity] = outcome
            tag = "solved"
        if "direction" not in outcome:
            rows.append([label] + [float("nan")] * (len(RESULT_COLUMNS) - 1))
            print("[{0}/{1}] {2} ({3}) ({4}) terminal status={5}: {6}".format(
                index, total, label, stem, tag,
                outcome.get("_status", "error"), outcome.get("_error", "")),
                flush=True)
            continue
        directions.append(outcome["direction"])
        normalization_flags.append(outcome["normalization_applied"])
        rows.append([label, outcome["rp"], outcome["ev"], outcome["eev"], outcome["ws"],
                     outcome["vss_percent"], outcome["evpi_percent"]])
        print("[{0}/{1}] {2} ({3}) ({10}) RP={4} EV={5} EEV={6} WS={7} VSS(%)={8} EVPI(%)={9}".format(
            index, total, label, stem,
            format_metric(outcome["rp"], 6), format_metric(outcome["ev"], 6),
            format_metric(outcome["eev"], 6), format_metric(outcome["ws"], 6),
            format_metric(outcome["vss_percent"], 4),
            format_metric(outcome["evpi_percent"], 4), tag), flush=True)
    write_results_workbook(rows, directions, normalization_flags)


if __name__ == "__main__":
    main()
