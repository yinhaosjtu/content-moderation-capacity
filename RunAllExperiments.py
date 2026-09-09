
import os
import subprocess
import sys
import time

PYTHON = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))

MODULES = [
    "SensitivityAnalysisExperiments.experiment_m1_anatomy",
    "SensitivityAnalysisExperiments.experiment_correlation",
    "SensitivityAnalysisExperiments.experiment_joint_value",
    "SensitivityAnalysisExperiments.experiment_vss_and_evpi",
    "SensitivityAnalysisExperiments.experiment_convergence",
    "ComparativeExperiments.comparative_experiment",
    "ComparativeExperiments.ablation_experiment",
    "ComparativeExperiments.scalability_experiment",
]

LOG_FILE = os.path.join(ROOT, "RunAllExperiments.log")


def log(handle, msg):
    line = "[{0}] {1}".format(time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def main():
    try:
        import gurobipy
    except ImportError:
        sys.exit("ERROR: gurobipy not found in {0}. Launch with: "
                 "py -3.10 RunAllExperiments.py".format(PYTHON))
    results = []
    with open(LOG_FILE, "a", encoding="utf-8") as handle:
        log(handle, "==== all-experiments serial run start ({0}) ====".format(
            len(MODULES)))
        for mod in MODULES:
            log(handle, "START {0}".format(mod))
            t0 = time.time()
            proc = subprocess.Popen(
                [PYTHON, "-u", "-m", mod],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                handle.write(line)
                handle.flush()
            proc.wait()
            dt = time.time() - t0
            ok = proc.returncode == 0
            results.append((mod, ok, dt, proc.returncode))
            log(handle, "END   {0}  {1}  ({2:.0f}s, rc={3})".format(
                mod, "OK" if ok else "FAILED", dt, proc.returncode))
        log(handle, "==== summary ====")
        for mod, ok, dt, rc in results:
            log(handle, "  {0:52s} {1:7s} {2:8.0f}s rc={3}".format(
                mod, "OK" if ok else "FAILED", dt, rc))
        n_ok = sum(1 for _, ok, _, _ in results if ok)
        log(handle, "==== done: {0}/{1} succeeded ====".format(n_ok, len(results)))


if __name__ == "__main__":
    main()
