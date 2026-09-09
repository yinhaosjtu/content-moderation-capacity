import json
import sys
import time

import Solvers.solver_heuristic as heuristic_module


def main():
    instance_path = sys.argv[1]
    time_limit = float(sys.argv[2])
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    with open(instance_path, "r", encoding="utf-8") as handle:
        instance = json.load(handle)

    params = {
        "time_limit": time_limit,
        "threads": threads,
        "output": False,
        "adaptive": True,
        "lp_backend": "scipy",
    }

    started = time.perf_counter()
    raw = heuristic_module.solve_instance(instance, params)
    elapsed = time.perf_counter() - started

    payload = {
        "objective": raw.get("objective"),
        "gap": raw.get("gap"),
        "time": raw.get("time", elapsed),
        "status": raw.get("status"),
    }
    sys.stdout.write("\n__RESULT__" + json.dumps(payload) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
