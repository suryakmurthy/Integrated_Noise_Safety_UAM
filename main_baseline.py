# main_baseline.py
#
# Driver for the shared-scheduling-protocol heuristic baselines
# (D2MAV_A/runner_baseline.py). Deliberately separate from main.py: there is
# no Agent, no gin config, no weights file, no training loop here -- this
# just runs a pool of BaselineRunner actors for N episodes each and
# aggregates the resulting metrics (LOS counts, halting time, travel time).
#
# Usage:
#   python main_baseline.py --heuristic round_robin --iterations 50 --num_workers 4
#   python main_baseline.py --heuristic csma_cd --iterations 50 --num_workers 4

import ray
import os
import glob
import argparse
import platform
import json
import numpy as np

from D2MAV_A.runner_baseline import BaselineRunner

os.environ["PYTHONPATH"] = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("--cluster", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--heuristic", choices=["round_robin", "csma_cd"], default="round_robin")
parser.add_argument("--iterations", type=int, default=50)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--max_steps", type=int, default=1024)
parser.add_argument("--simdt", type=int, default=4)
parser.add_argument(
    "--speeds",
    type=int,
    nargs=3,
    default=[5, 0, 60],
    help="min/hold/cruise speed in knots -- matches conf/config_test.gin's Driver.speeds (UAM-scale, not commercial)",
)
parser.add_argument(
    "--scenario_file",
    type=str,
    default="scenarios/generated_scenarios/austin_env_full_ver_4.scn",
)
parser.add_argument("--config_file", type=str, default="settings.cfg")
parser.add_argument("--LOS", type=int, default=150)
parser.add_argument("--dGoal", type=int, default=500)
parser.add_argument("--intruderThreshold", type=int, default=2500)
parser.add_argument("--max_alt", type=int, default=3000)
parser.add_argument("--min_alt", type=int, default=1000)
parser.add_argument("--alt_level_separation", type=int, default=500)
parser.add_argument("--gui", action="store_true")
parser.add_argument("--no_gui_realtime", action="store_true", help="Disable real-time pacing even with --gui (runs fast, won't be watchable)")
parser.add_argument("--gui_speed", type=float, default=1.0, help="Playback speed multiplier with --gui (e.g. 4.0 = 4x faster than real-time, still watchable)")
parser.add_argument("--debug_heuristic", action="store_true", help="Print per-step round-robin/CSMA-CD state")
parser.add_argument(
    "--eval_scenario_dir",
    type=str,
    default=None,
    help="If set, deterministically run every .scn file in this directory exactly once "
         "(e.g. scenarios/eval), instead of --iterations random draws from --scenario_file. "
         "Lets a baseline run and an RL eval run be directly paired scenario-for-scenario.",
)
args = parser.parse_args()


def main():
    if args.cluster:
        ray.init(address=os.environ["ip_head"])
    else:
        if platform.release() == "Darwin":
            ray.init(_node_ip_address="0.0.0.0", local_mode=args.debug)
        else:
            ray.init(local_mode=args.debug)
    print(ray.cluster_resources())

    working_directory = os.getcwd()

    workers = {
        i: BaselineRunner.remote(
            i,
            heuristic=args.heuristic,
            scenario_file=args.scenario_file,
            working_directory=working_directory,
            config_file=args.config_file,
            max_steps=args.max_steps,
            simdt=args.simdt,
            speeds=args.speeds,
            LOS=args.LOS,
            dGoal=args.dGoal,
            intruderThreshold=args.intruderThreshold,
            max_alt=args.max_alt,
            min_alt=args.min_alt,
            alt_level_separation=args.alt_level_separation,
            gui=args.gui,
            gui_realtime=not args.no_gui_realtime,
            gui_speed=args.gui_speed,
            debug=args.debug_heuristic,
        )
        for i in range(args.num_workers)
    }

    path_results = f"results/baseline_{args.heuristic}"
    os.makedirs(path_results, exist_ok=True)

    all_los_counter = []
    all_los_events = []
    all_los_same_route = []
    all_los_diff_route = []
    all_nmacs = []
    all_nmac_time = []
    all_avg_halting_time = []
    all_avg_midair_halting_time = []
    all_avg_travel_time = []
    all_total_spawned = []
    all_total_completed = []
    all_max_noise_increase = []
    all_mean_noise_increase = []
    per_scenario_results = []  # only populated in --eval_scenario_dir mode

    def record(data):
        all_los_counter.append(data["los_counter"])
        all_los_events.append(data["los_events"])
        all_los_same_route.append(data["los_same_route_events"])
        all_los_diff_route.append(data["los_diff_route_events"])
        all_nmacs.append(data["nmacs"])
        all_nmac_time.append(data["nmac_time"])
        all_avg_halting_time.append(data["avg_halting_time"])
        all_avg_midair_halting_time.append(data["avg_midair_halting_time"])
        all_avg_travel_time.append(data["avg_travel_time"])
        all_total_spawned.append(data["total_ac_spawned"])
        all_total_completed.append(data["total_ac_completed"])
        all_max_noise_increase.append(data["max_noise_increase"])
        all_mean_noise_increase.append(data["mean_noise_increase"])

    def print_progress(label):
        print(f"     {label} ({args.heuristic})     ")
        print("|------------------------------|")
        print(f"| Mean LOS Counter:      {np.mean(all_los_counter):.2f}  |")
        print(f"| Mean LOS Events:       {np.mean(all_los_events):.2f}  |")
        print(f"|   same-route:          {np.sum(all_los_same_route):.0f}  |")
        print(f"|   different-route:     {np.sum(all_los_diff_route):.0f}  |")
        print(f"| NMACs (RL-comparable): {np.sum(all_nmacs):.0f}  |")
        print(f"| NMAC time:             {np.sum(all_nmac_time):.1f} s  |")
        print(f"| Mean Avg Halting Time: {np.mean(all_avg_halting_time):.2f}  |")
        print(f"| Mean Midair Halting Time: {np.mean(all_avg_midair_halting_time):.2f}  |")
        print(f"| Mean Avg Travel Time:  {np.mean(all_avg_travel_time):.2f}  |")
        completed_pct = 100.0 * np.sum(all_total_completed) / max(1, np.sum(all_total_spawned))
        print(f"| Completed:  {np.sum(all_total_completed):.0f}/{np.sum(all_total_spawned):.0f} ({completed_pct:.1f}%)  |")
        print(f"| Mean Noise Increase:   {np.mean(all_mean_noise_increase):.2f} dB  |")
        print(f"| Max Noise Increase:    {np.max(all_max_noise_increase):.2f} dB  |")
        print("|------------------------------|")

    def save_arrays():
        np.save(f"{path_results}/los_counter.npy", np.array(all_los_counter))
        np.save(f"{path_results}/los_events.npy", np.array(all_los_events))
        np.save(f"{path_results}/avg_halting_time.npy", np.array(all_avg_halting_time))
        np.save(f"{path_results}/avg_midair_halting_time.npy", np.array(all_avg_midair_halting_time))
        np.save(f"{path_results}/avg_travel_time.npy", np.array(all_avg_travel_time))
        np.save(f"{path_results}/mean_noise_increase.npy", np.array(all_mean_noise_increase))
        np.save(f"{path_results}/max_noise_increase.npy", np.array(all_max_noise_increase))

    if args.eval_scenario_dir:
        # Deterministic sweep: exactly one episode per .scn file, so this
        # can be paired scenario-for-scenario against an RL eval run over
        # the same directory (run main.py once per file the same way).
        scenario_queue = sorted(glob.glob(os.path.join(args.eval_scenario_dir, "*.scn")))
        print(f"Sweeping {len(scenario_queue)} scenarios from {args.eval_scenario_dir}")

        worker_list = list(workers.values())
        in_flight = {}  # ray future -> scenario path
        for w in worker_list:
            if not scenario_queue:
                break
            path = scenario_queue.pop(0)
            in_flight[w.run_one_iteration.remote(scenario_file_override=path)] = path

        completed = 0
        total = len(in_flight) + len(scenario_queue)
        while in_flight:
            done_id, _ = ray.wait(list(in_flight.keys()), num_returns=1)
            fut = done_id[0]
            path = in_flight.pop(fut)
            data, worker_id = ray.get(ray.get(fut))
            record(data)
            per_scenario_results.append({"scenario_file": path, **data})
            completed += 1
            print(
                f"     {completed}/{total} scenarios complete ({args.heuristic})     \n"
                f"| {path}: LOS={data['nmacs']} max_noise={data['max_noise_increase']:.2f} "
                f"mean_travel={data['avg_travel_time']:.1f} mean_airborne={data['mean_airborne_count']:.1f} |"
            )

            if scenario_queue:
                next_path = scenario_queue.pop(0)
                in_flight[workers[worker_id].run_one_iteration.remote(scenario_file_override=next_path)] = next_path

            if completed % max(1, total // 10) == 0 or completed == total:
                print_progress(f"{completed}/{total} scenarios complete")
                save_arrays()

        with open(f"{path_results}/per_scenario_results.json", "w") as f:
            json.dump(per_scenario_results, f, indent=4)

        # Properly pooled aggregate: every individual event from every
        # scenario, concatenated into one flat list per metric, THEN a
        # single mean/std over the pooled data -- not a mean of 20
        # already-averaged scenario-level numbers. That "mean of means"
        # is what the scenario-level summary above computes, and it can
        # understate variance and implicitly weight every scenario
        # equally regardless of how many aircraft/events it actually
        # contained. This is the statistic that should actually get
        # reported; the scenario-level mean is left in place only as a
        # quick sanity-check number.
        pooled_halting_times = []
        pooled_midair_halting_times = []
        pooled_travel_times = []
        pooled_nmac_event_lengths = []
        pooled_total_levels_climbed = []  # one value PER AIRCRAFT (summed across
                                           # all its climb events), not per event --
                                           # an aircraft that climbed 2 levels total
                                           # contributes one entry of 2.0, whether
                                           # that was one 2-level climb or two
                                           # separate 1-level climbs
        pooled_noise_samples = []
        los_counts_per_scenario = []
        max_noise_per_scenario = []

        for entry in per_scenario_results:
            pooled_halting_times.extend(entry.get("halting_times", []))
            pooled_midair_halting_times.extend(entry.get("midair_halting_times", []))
            pooled_travel_times.extend(list(entry.get("full_travel_times", {}).values()))
            pooled_nmac_event_lengths.extend(entry.get("nmac_event_lengths", []))
            for ac_climb_events in entry.get("full_alt_adjustments", {}).values():
                pooled_total_levels_climbed.append(sum(ac_climb_events))  # 0.0 for aircraft that never climbed
            for route_samples in entry.get("avg_noise_increase", {}).values():
                pooled_noise_samples.extend(route_samples)
            los_counts_per_scenario.append(entry.get("nmacs", 0))
            max_noise_per_scenario.append(entry.get("max_noise_increase", 0.0))

        def mean_std(values):
            if not values:
                return {"mean": 0.0, "std": 0.0, "n": 0}
            arr = np.array(values, dtype=float)
            return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "n": int(len(arr))}

        pooled_summary = {
            "heuristic": args.heuristic,
            "n_scenarios": len(per_scenario_results),
            "los_events_per_scenario": mean_std(los_counts_per_scenario),  # LOS is inherently a per-scenario count
            "total_los_events": int(np.sum(los_counts_per_scenario)),
            "max_noise_increase_db": {"max_over_scenarios": float(np.max(max_noise_per_scenario)) if max_noise_per_scenario else 0.0},
            "noise_increase_db": mean_std(pooled_noise_samples),          # pooled over every route/step/scenario
            "travel_time_s": mean_std(pooled_travel_times),               # pooled over every aircraft/scenario
            "halting_time_s": mean_std(pooled_halting_times),             # pooled over every halt event/scenario
            "midair_halting_time_s": mean_std(pooled_midair_halting_times),
            "total_levels_climbed_per_aircraft": mean_std(pooled_total_levels_climbed),  # pooled over every aircraft/scenario
            "nmac_event_length_s": mean_std(pooled_nmac_event_lengths),
        }

        with open(f"{path_results}/pooled_summary.json", "w") as f:
            json.dump(pooled_summary, f, indent=4)
        print(f"\nSaved pooled (not mean-of-means) summary to {path_results}/pooled_summary.json")
        print(json.dumps(pooled_summary, indent=2))

    else:
        runner_sims = [workers[wid].run_one_iteration.remote() for wid in workers.keys()]

        for i in range(args.iterations):
            done_id, runner_sims = ray.wait(runner_sims, num_returns=args.num_workers)
            results = ray.get(done_id)

            for result in results:
                data, _worker_id = ray.get(result)
                record(data)

            print_progress(f"Iteration {i} Complete")
            save_arrays()

            if i < args.iterations - 1:
                runner_sims = [workers[wid].run_one_iteration.remote() for wid in workers.keys()]

    summary = {
        "heuristic": args.heuristic,
        "mode": "eval_scenario_dir" if args.eval_scenario_dir else "random_iterations",
        "eval_scenario_dir": args.eval_scenario_dir,
        "n_episodes": len(all_los_counter),
        "iterations": args.iterations,
        "mean_los_counter": float(np.mean(all_los_counter)),
        "mean_los_events": float(np.mean(all_los_events)),
        "total_los_same_route_events": int(np.sum(all_los_same_route)),
        "total_los_diff_route_events": int(np.sum(all_los_diff_route)),
        "total_nmacs": int(np.sum(all_nmacs)),
        "total_nmac_time": float(np.sum(all_nmac_time)),
        "mean_avg_halting_time": float(np.mean(all_avg_halting_time)),
        "mean_avg_midair_halting_time": float(np.mean(all_avg_midair_halting_time)),
        "mean_avg_travel_time": float(np.mean(all_avg_travel_time)),
        "total_ac_spawned": int(np.sum(all_total_spawned)),
        "total_ac_completed": int(np.sum(all_total_completed)),
        "pct_completed": float(100.0 * np.sum(all_total_completed) / max(1, np.sum(all_total_spawned))),
        "mean_noise_increase_db": float(np.mean(all_mean_noise_increase)),
        "max_noise_increase_db": float(np.max(all_max_noise_increase)),
    }
    with open(f"{path_results}/summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("Final summary:", summary)


if __name__ == "__main__":
    main()