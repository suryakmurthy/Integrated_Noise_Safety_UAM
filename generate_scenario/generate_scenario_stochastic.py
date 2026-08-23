import os
import json
import random
from copy import deepcopy


def _sec_to_timestr(tsec: int) -> str:
    """Convert seconds-from-start to BlueSky scenario timestamp."""
    if tsec < 0:
        tsec = 0
    hh = tsec // 3600
    mm = (tsec % 3600) // 60
    ss = tsec % 60
    return f"{hh:02}:{mm:02}:{ss:02}.00"


def _sample_route_counts_gaussian(
    base_demand: dict,
    rng: random.Random,
    gaussian_sigma_frac: float = 0.25,
    min_count: int = 0,
    max_count: int = 15
) -> dict:
    out = {}
    for r, base in base_demand.items():
        base = max(0, int(base))
        sigma = gaussian_sigma_frac * base
        cnt = int(round(rng.gauss(base, sigma)))
        if max_count is not None:
            cnt = min(cnt, max_count)
        cnt = max(min_count, cnt)
        out[r] = cnt
    return out


def generate_scenario_austin_stochastic(
    out_folder: str,
    demand_dict_path: str,
    route_dict_path: str,
    n_scenarios: int = 10,
    seed_base: int = 1234,

    # Demand randomness
    gaussian_sigma_frac: float = 0.25,

    # Takeoff-time randomness
    interdep_mode: str = "exponential",        # "exponential" | "uniform"
    mean_interdep_sec: float = 10.0,           # mean spacing between departures at same start waypoint
    uniform_interdep_range=(5.0, 15.0),        # if interdep_mode="uniform"
    min_sep_sec: int = 1,                      # hard minimum separation at same start waypoint
    initial_offset_range=(0, 30),              # initial random offset per starting waypoint

    # Optional extra jitter added to each takeoff time (after separation enforcement)
    takeoff_jitter_sec: int = 0                # e.g. 2 means add randint(-2,+2)
):
    os.makedirs(out_folder, exist_ok=True)

    with open(demand_dict_path, "r") as f:
        base_demand = json.load(f)

    with open(route_dict_path, "r") as f:
        route_dict = json.load(f)

    route_names = list(route_dict.keys())

    for sc_idx in range(n_scenarios):
        rng = random.Random(seed_base + sc_idx)

        # 1) Sample route counts for THIS scenario
        sampled_counts = _sample_route_counts_gaussian(
            base_demand=base_demand,
            rng=rng,
            gaussian_sigma_frac=gaussian_sigma_frac,
            min_count=0,
            max_count=15
        )

        # 2) Build per-start-waypoint schedule using random inter-departure gaps
        # Group routes by starting waypoint (so we can enforce separation there)
        start_to_routes = {}
        for r in route_names:
            wpts = route_dict[r]
            starting_wpt = wpts[0] + wpts[1] + "1"
            start_to_routes.setdefault(starting_wpt, []).append(r)

        # For each start waypoint, we will "emit" departures for its routes in a randomized round-robin.
        # next_available_time enforces min separation.
        next_available_time = {}
        for start_wpt in start_to_routes:
            lo, hi = initial_offset_range
            next_available_time[start_wpt] = int(rng.uniform(lo, hi))

        # Create a list of (time_str, command_str) entries
        entries = []

        # Helper to sample an interdeparture gap (>= min_sep_sec)
        def sample_gap_sec() -> int:
            if interdep_mode == "exponential":
                gap = rng.expovariate(1.0 / max(1e-6, mean_interdep_sec))
            elif interdep_mode == "uniform":
                a, b = uniform_interdep_range
                gap = rng.uniform(a, b)
            else:
                raise ValueError(f"Unknown interdep_mode: {interdep_mode}")

            gap_i = int(round(gap))
            return max(min_sep_sec, gap_i)

        total_aircraft_num = 0

        # For each starting waypoint, schedule all aircraft that originate there
        for start_wpt, routes in start_to_routes.items():
            # Build a bag of route "tokens" for this start waypoint
            # Example: if route A has 3 and route B has 1 => tokens [A,A,A,B]
            tokens = []
            for r in routes:
                tokens.extend([r] * int(sampled_counts.get(r, 0)))

            # Randomize departure order among those routes
            rng.shuffle(tokens)

            # Now assign times sequentially at this start waypoint with random gaps
            t = next_available_time[start_wpt]
            for r in tokens:
                wpts = route_dict[r]
                first_wpt = wpts[0] + wpts[1] + "1"
                last_wpt = wpts[-2] + wpts[-1] + "2"

                # optional jitter (still keep nonnegative, but note: this can violate min_sep slightly if large)
                tj = 0
                if takeoff_jitter_sec > 0:
                    tj = rng.randint(-takeoff_jitter_sec, takeoff_jitter_sec)

                takeoff_t = max(0, t + tj)
                time_str = _sec_to_timestr(takeoff_t)

                plane = f"P{r}{total_aircraft_num}"
                # Keep your original command format (time + ">CMD ...")
                entries.append((time_str, f">CRE {plane},EC35,{first_wpt},0,0\n"))
                entries.append((time_str, f">ORIG {plane} {first_wpt}\n"))
                entries.append((time_str, f">DEST {plane} {last_wpt}\n"))
                entries.append((time_str, f">SPD {plane} 40\n"))
                entries.append((time_str, f">ALT {plane} 800\n"))

                for idx in range(0, len(wpts) - 1):
                    waypoint_1 = wpts[idx] + wpts[idx + 1] + "1"
                    waypoint_2 = wpts[idx] + wpts[idx + 1] + "2"
                    entries.append((time_str, f">ADDWPT {plane} {waypoint_1} 800 40\n"))
                    entries.append((time_str, f">ADDWPT {plane} {waypoint_2} 800 40\n"))

                entries.append((time_str, f">{plane} VNAV on \n"))

                # advance time at this starting waypoint
                t += sample_gap_sec()
                total_aircraft_num += 1

        # 3) Sort by time and write scenario file
        entries.sort(key=lambda x: x[0])

        scn_path = os.path.join(out_folder, f"austin_env_full_single_intersection_sc{sc_idx:03d}.scn")
        with open(scn_path, "w") as f:
            f.write("00:00:00.00>TRAILS ON \n\n")
            f.write("00:00:00.00>PAN 30.29828195311632 -97.92645392342473 \n\n")

            for tstr, cmd in entries:
                # Your original format was: f.write(entry[0] + entry[1])
                # where entry[0] already included "00:..". We keep that:
                f.write(f"{tstr}{cmd}")

        print(f"Wrote scenario: {scn_path}  (aircraft: {total_aircraft_num})")


# Example usage:
generate_scenario_austin_stochastic(
    out_folder="/home/suryamurthy/UT_Autonomous_Group/prev_repos/ULI_noise_aware_agent/scenarios/generated_scenarios/stochastic_scenarios",
    demand_dict_path="/home/suryamurthy/UT_Autonomous_Group/prev_repos/ULI_noise_aware_agent/D2MAV_A/route_demand_dict.json",
    route_dict_path="/home/suryamurthy/UT_Autonomous_Group/prev_repos/ULI_noise_aware_agent/D2MAV_A/route_info_dict.json",
    n_scenarios=20,
    seed_base=2026,
    interdep_mode="uniform",
    uniform_interdep_range=(90, 150),
    mean_interdep_sec=90.0,
    min_sep_sec=120,
    initial_offset_range=(0, 60),
    takeoff_jitter_sec=0
)
