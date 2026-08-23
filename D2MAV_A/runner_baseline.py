# D2MAV_A/runner_baseline.py
#
# Standalone shared-scheduling-protocol (SSP) heuristic baseline for the UAM
# noise/safety environment. This does NOT use the D2MAV-A RL policy -- it is
# a pure heuristic controller, meant as a baseline comparison point.
#
# Ported (by hand, not merged) from an older speed/halting-based SSP runner.
# The old logic controlled separation purely through ground-speed halting;
# this repo's action space is altitude-only (ascend/maintain/descend across
# discrete flight levels), so both heuristics were re-designed around
# altitude-slot assignment instead of speed. Conceptually:
#
#   1. "round_robin": treats each intersection's approach as a queue.
#      Aircraft are granted altitude "slots" (one aircraft per discrete
#      altitude level) in FIFO/round-robin order as they arrive. Aircraft
#      beyond the number of available levels are halted (speed = 0) until a
#      slot frees up (i.e. an active aircraft clears the intersection).
#
#   2. "csma_cd": fully decentralized, no queue. When an aircraft detects a
#      conflict (its remaining flight path geometrically intersects another
#      aircraft's, within range, at the same altitude level), it halts and
#      waits a random backoff (in simulation steps). When the backoff
#      expires, it checks whether a nearby altitude level is free; if so it
#      commits to that level, otherwise it picks a new random backoff and
#      waits again -- the classic CSMA/CD collision -> random backoff -> retry
#      pattern.
#
# STATUS: first-pass implementation. It reuses BlueSky/traffic-manager
# plumbing copied from D2MAV_A/runner.py (scenario loading, intersection
# tracking, goal/LOS bookkeeping) but has not been run end-to-end yet.
# Treat this as a starting point to debug against your actual BlueSky
# environment, not a finished/verified implementation.

import numpy as np
import random
import os
import glob
import json
import math
import yaml
import pickle
import ray
import time
from collections import deque
from itertools import groupby

from bluesky.tools import geo
from shapely.geometry import LineString
from shapely.geometry.multilinestring import MultiLineString
from pyproj import Transformer

from D2MAV_A.qatc import TrafficManager, VehicleHelper, load_routes

FILE_PREFIX = str(os.path.dirname(__file__))
TOWER_CONFIG_FILE = FILE_PREFIX + "/Austin_towers.yaml"
with open(TOWER_CONFIG_FILE, "r") as f:
    tower_config = yaml.load(f, Loader=yaml.Loader)

# Same file D2MAV_A/runner.py uses -- ambient dB baseline per route_section
# and per intersection, needed to compute noise INCREASE (not raw noise) the
# same way the RL pipeline does, so the two are directly comparable.
with open(FILE_PREFIX + "/ambient_noise_dict.json", "r") as f:
    ambient_noise_dict = json.load(f)

# NOTE: matches D2MAV_A/runner.py's route-data file. Adjust the path if your
# baseline scenarios use a different pickle.
with open("new_route_data.pkl", "rb") as file:
    route_data = pickle.load(file)


@ray.remote
class BaselineRunner:
    """
    Pure-heuristic baseline runner. Mirrors the BlueSky plumbing of
    D2MAV_A/runner.py's Runner class, but replaces the RL policy with one of
    two shared-scheduling heuristics and drops all PPO/reward-shaping
    machinery -- there is no agent here, and nothing is trained.

    This is a Ray actor (like D2MAV_A/runner.py's Runner) so it plugs into
    the same worker-pool pattern main.py uses -- see main_baseline.py.
    """

    import bluesky as bs

    def __init__(
        self,
        actor_id,
        heuristic="round_robin",       # "round_robin" or "csma_cd"
        max_steps=1024,
        speeds=[5, 0, 60],  # matches conf/config_test.gin's Driver.speeds --
                             # NOT the [5,0,220] fallback default in
                             # D2MAV_A/runner.py's constructor, which is
                             # never actually used there (main.py's Driver
                             # always overrides it) and is scaled for
                             # commercial aircraft, not this UAM network.
                             # Running at 220kt instead of 60kt meant every
                             # spacing buffer below was implicitly sized for
                             # ~4x too slow an aircraft, so halts couldn't
                             # actually stop aircraft in time.
        simdt=1,
        scenario_file=None,
        working_directory=None,
        LOS=10,
        dGoal=100,
        intruderThreshold=750,
        config_file=None,
        gui=False,
        max_alt=3000,
        min_alt=1000,
        alt_level_separation=500,
        backoff_min_steps=1,
        backoff_max_steps=20,
        same_route_spacing=None,  # meters; minimum in-trail gap between
                                   # aircraft sharing the same route section.
                                   # Defaults to LOS + a buffer if not set.
        alt_arrival_tolerance=100,  # ft; how close to its assigned altitude
                                     # an aircraft must be before it's
                                     # cleared to cross -- it must finish
                                     # climbing/descending first, not do it
                                     # while already crossing.
        holding_spacing=None,  # meters; minimum gap between ANY two
                                # aircraft converging on the same
                                # intersection while holding/queued, not
                                # just same-route ones. Bigger than
                                # same_route_spacing and triggers earlier,
                                # since a halted aircraft can't back away
                                # once it's already stopped.
        debug=False,
        queue_radius=8100,
        max_slot_hold_steps=150,  # real rotation: once a route_section has
                                   # held an intersection slot this long,
                                   # and something else is actually
                                   # waiting, force it back into the queue
                                   # so it can't starve other route
                                   # sections converging on the same
                                   # intersection forever. Only evicted at
                                   # a safe gap -- never while one of its
                                   # aircraft is mid-crossing.
        gui_realtime=True,  # when gui=True, sleep between steps so the
                             # simulation runs close to wall-clock speed and
                             # the BlueSky client has time to connect and
                             # actually show it, instead of finishing before
                             # you can pick the node  # meters; aircraft only join a round-robin queue
                             # once within this distance of the intersection,
                             # not for their entire flight to it (see bug notes
                             # in _round_robin_actions)
        gui_speed=1.0,  # playback speed multiplier when gui_realtime=True --
                         # 1.0 = real-time (1 sim-second per real second),
                         # 4.0 = 4x faster than real-time but still watchable,
                         # unlike --no_gui_realtime which removes pacing
                         # entirely and runs too fast to actually see.
    ):
        self.id = actor_id
        assert heuristic in ("round_robin", "csma_cd"), heuristic
        self.heuristic = heuristic

        self.scen_file = scenario_file
        self.working_directory = working_directory
        self.speeds = np.array(speeds)
        self.simdt = simdt
        self.max_steps = max_steps
        self.LOS = LOS
        self.dGoal = dGoal
        self.intruderThreshold = intruderThreshold
        self.gui = gui

        self.min_alt = min_alt
        self.max_alt = max_alt
        self.alt_level_separation = alt_level_separation
        # Discrete altitude levels available in the airspace, e.g. [1000,1500,...,3000]
        n_levels = int(round((max_alt - min_alt) / alt_level_separation)) + 1
        self.alt_levels = [min_alt + i * alt_level_separation for i in range(n_levels)]

        self.backoff_min_steps = backoff_min_steps
        self.backoff_max_steps = backoff_max_steps
        # LOS+50 was tried and confirmed too tight: against a FULLY
        # STATIONARY target (queued/frozen aircraft ahead), the trailing
        # aircraft's closing speed is its entire cruise speed, not some
        # smaller relative velocity -- there wasn't enough real distance
        # left to decelerate before crossing into LOS. LOS+200 gives real
        # stopping margin for that case while still being tighter than the
        # original LOS+250.
        self.same_route_spacing = same_route_spacing if same_route_spacing is not None else LOS + 200
        self.alt_arrival_tolerance = alt_arrival_tolerance
        self.holding_spacing = holding_spacing if holding_spacing is not None else LOS + 600
        self.debug = debug
        self.queue_radius = queue_radius
        self.max_slot_hold_steps = max_slot_hold_steps
        self.gui_realtime = gui_realtime
        self.gui_speed = gui_speed

        self.epsg_proj = "epsg:2163"
        self.epsg_from = "epsg:4326"
        self.transformer = Transformer.from_crs(self.epsg_from, self.epsg_proj, always_xy=True)

        self.intersection_radius = 2700

        # Single-event noise model, identical to D2MAV_A/runner.py's, so
        # noise impact is directly comparable between the RL policy and
        # this heuristic: dB(alt_ft) = a_0 + a_1*log10(alt) + a_2*log10(alt)^2
        self.a_0 = 88.09
        self.a_1 = 3.21
        self.a_2 = -2.62
        self.ambient_noise_level = ambient_noise_dict

        if "SIMDT" not in os.environ.keys():
            os.environ["SIMDT"] = "{}".format(self.simdt)

        self.step_counter = 0
        self.episode_done = True

        self.create_traffic_manager()

        if self.gui:
            self.bs.init(mode="sim", configfile=self.working_directory + "/" + config_file)
            self.bs.net.connect()
        else:
            self.bs.init(mode="sim", detached=True, configfile=self.working_directory + "/" + config_file)

    # ------------------------------------------------------------------
    # BlueSky / traffic-manager plumbing (adapted from D2MAV_A/runner.py)
    # ------------------------------------------------------------------
    def create_traffic_manager(self):
        route_linestrings = {}
        for route_id, gps_wp_list in route_data.items():
            rtemp = []
            for item in gps_wp_list:  # item is a tuple of (lon, lat)
                x, y = self.transformer.transform(item[0], item[1])
                rtemp.append((x, y))
            route_linestrings[route_id] = LineString(rtemp)
        self.traffic_manager = TrafficManager(tower_config)
        self.vehicle_helpers = {}
        self.routes_loaded = load_routes(tower_config, self.traffic_manager, route_linestrings)

    def meters_to_feet(self, meters):
        return meters * 3.28084

    def _get_intersection_radius_m(self, intersection_id):
        """
        Intersection.__init__ computes its circular boundary (location +
        200m buffer) internally via create_shapely_objects(), but only
        keeps the resulting shapely region_shape/region_ring -- it never
        saves the radius itself as an attribute (there is no
        Intersection.radius; the town-config 'radius' field feeding into
        that computation is a second lat/lon point, not a scalar, and
        isn't retained either). Derive it from region_shape's bounding box
        (a circle's bbox width is its diameter) rather than assuming a raw
        .radius field exists anywhere. Cached since it never changes for a
        given intersection.
        """
        if not hasattr(self, "_intersection_radius_cache"):
            self._intersection_radius_cache = {}
        if intersection_id not in self._intersection_radius_cache:
            intersection = self.traffic_manager.intersections[intersection_id]
            minx, miny, maxx, maxy = intersection.region_shape.bounds
            self._intersection_radius_cache[intersection_id] = (maxx - minx) / 2
        return self._intersection_radius_cache[intersection_id]

    def _stopping_distance_margin(self, idx):
        """
        Conservative extra buffer (meters), on top of intersection_radius +
        LOS, added when deciding when an ungranted aircraft must actually
        halt. Without this, the plain LOS-sized buffer left barely any
        room to actually decelerate before crossing the boundary --
        confirmed directly via trace: aircraft were getting the halt
        command only 1-2 steps before physically entering the
        intersection, nowhere near enough time to slow from cruise speed.
        v^2 / (2a) with a conservative, gentle assumed deceleration rate --
        scales with the aircraft's OWN current speed, so a slower aircraft
        (already partway through decelerating) doesn't get an unnecessarily
        large buffer.
        """
        current_speed = self.bs.traf.cas[idx]  # m/s
        if current_speed <= 0:
            return 0.0
        assumed_decel = 1.5  # m/s^2 -- conservative/gentle
        return (current_speed ** 2) / (2 * assumed_decel)

    def _pairwise_distance_matrix(self):
        n_ac = self.bs.traf.lat.shape[0]
        if n_ac == 0:
            return np.zeros((0, 0))
        return (
            geo.kwikdist_matrix(
                np.repeat(self.bs.traf.lat, n_ac),
                np.repeat(self.bs.traf.lon, n_ac),
                np.tile(self.bs.traf.lat, n_ac),
                np.tile(self.bs.traf.lon, n_ac),
            ).reshape(n_ac, n_ac)
            * geo.nm
        )

    def _enforce_same_route_spacing(self, d):
        """
        Aircraft sharing the same current route section are flying the
        identical path (same entry, same exit) -- round-robin/CSMA-CD don't
        manage them at all, since neither heuristic treats same-route
        aircraft as being in conflict with each other. But they still need
        in-trail separation or the trailing one can run up on the aircraft
        ahead.

        Ordered by distance remaining to the END OF THE CURRENT SEGMENT
        (via next_intersection's location) -- NOT geometries.geoms[i].length,
        which is total remaining distance to the aircraft's FINAL
        destination across all FUTURE hops. That was the actual bug: two
        aircraft on the identical current segment right now can have wildly
        different total-trip-remaining values once their paths diverge
        after this segment (one continuing for several more hops, the other
        on its last one) -- so "ahead by total remaining distance" doesn't
        mean "ahead on THIS segment." Distance to the shared segment's own
        endpoint is a purely local, apples-to-apples comparison regardless
        of what either aircraft does afterward. Applies on top of whatever
        the active heuristic already decided -- this only ever ADDS halts,
        never removes one.

        Also checks a step ahead: a trailing aircraft still on its PRIOR
        segment, about to merge onto exactly the segment a leading aircraft
        is already on, gets the same protection -- not just aircraft
        already sharing the current segment. Without this, an aircraft
        approaching at full cruise speed gets zero warning about a queue
        already built up on the segment it's about to join, since the
        match on current_route_section alone doesn't apply until the exact
        step it actually merges -- which can already be too close.
        """
        n_ac = self.bs.traf.lat.shape[0]
        for i in range(n_ac):
            id_i = self.bs.traf.id[i]
            vh_i = self.vehicle_helpers.get(id_i)
            if vh_i is None or vh_i.current_route_section is None:
                continue
            target_i = vh_i.next_intersection
            if target_i is None or target_i not in self.traffic_manager.intersections:
                continue
            lat_t, lon_t = self.traffic_manager.intersections[target_i].location
            dist_i = geo.kwikdist(self.bs.traf.lat[i], self.bs.traf.lon[i], lat_t, lon_t) * geo.nm
            next_rs_i = getattr(vh_i, "next_route_section", None)

            for j in range(n_ac):
                if i == j:
                    continue
                id_j = self.bs.traf.id[j]
                vh_j = self.vehicle_helpers.get(id_j)
                if vh_j is None or vh_j.current_route_section is None:
                    continue

                if vh_j.current_route_section == vh_i.current_route_section:
                    # Already on the identical segment -- ahead = closer to
                    # its end.
                    dist_j = geo.kwikdist(self.bs.traf.lat[j], self.bs.traf.lon[j], lat_t, lon_t) * geo.nm
                    if dist_i <= dist_j:
                        continue
                elif next_rs_i is not None and next_rs_i == vh_j.current_route_section:
                    # i hasn't reached the shared segment yet, but is about
                    # to merge onto exactly the one j is already flying --
                    # definitionally behind j (which is already on it), no
                    # distance comparison needed to establish that.
                    pass
                else:
                    continue

                if d[i, j] < self.same_route_spacing:
                    if id_i not in self.action_override:
                        self.action_override.append(id_i)
                    self.same_route_halted_ids.add(id_i)
                    if id_i not in self.alt_override:
                        self.alt_override.append(id_i)
                    break

    def _halt_reason(self, id_, idx):
        """Diagnostic only: infers which mechanism halted a given aircraft this step."""
        if id_ in self.grounded:
            return "takeoff_hold"
        if id_ in getattr(self, "rr_halted_ids", set()):
            return "round_robin_queue"
        target = self.last_alt_targets.get(id_)
        if target is not None:
            current_ft = round(self.meters_to_feet(self.bs.traf.alt[idx]))
            if abs(current_ft - target) > self.alt_arrival_tolerance:
                return "altitude_gate"
        if id_ in self.same_route_halted_ids:
            return "same_route_spacing"
        if id_ in self.holding_halted_ids:
            return "holding_spacing"
        if id_ not in self.action_override:
            return "not_halted"
        return "unknown"

    def _record_pair_history(self, d):
        """
        Records a snapshot for every same-route-section pair, using the
        FINAL action_override for this step (called after every halt-
        deciding method has run) -- not a mid-pipeline snapshot. Recording
        this inside _enforce_same_route_spacing itself was misleading: that
        method runs before _enforce_holding_spacing, so an aircraft halted
        for holding-spacing reasons would show up as "not halted" in a
        snapshot taken before that decision existed yet, even though it's
        genuinely stationary by the time commands are actually issued.
        """
        n_ac = self.bs.traf.lat.shape[0]
        for i in range(n_ac):
            id_i = self.bs.traf.id[i]
            vh_i = self.vehicle_helpers.get(id_i)
            if vh_i is None or vh_i.current_route_section is None:
                continue
            target_i = vh_i.next_intersection
            if target_i is None or target_i not in self.traffic_manager.intersections:
                continue
            lat_t, lon_t = self.traffic_manager.intersections[target_i].location
            dist_i = geo.kwikdist(self.bs.traf.lat[i], self.bs.traf.lon[i], lat_t, lon_t) * geo.nm

            for j in range(i + 1, n_ac):
                id_j = self.bs.traf.id[j]
                vh_j = self.vehicle_helpers.get(id_j)
                if vh_j is None or vh_j.current_route_section != vh_i.current_route_section:
                    continue
                dist_j = geo.kwikdist(self.bs.traf.lat[j], self.bs.traf.lon[j], lat_t, lon_t) * geo.nm

                pair_key = tuple(sorted((id_i, id_j)))

                self.pair_history.setdefault(pair_key, deque(maxlen=60)).append(
                    {
                        "step": self.step_counter,
                        f"{id_i}_dist_to_segment_end": round(dist_i, 1),
                        f"{id_j}_dist_to_segment_end": round(dist_j, 1),
                        "gap": round(d[i, j], 1),
                        f"{id_i}_halted": id_i in self.action_override,
                        f"{id_j}_halted": id_j in self.action_override,
                        f"{id_i}_reason": self._halt_reason(id_i, i),
                        f"{id_j}_reason": self._halt_reason(id_j, j),
                        f"{id_i}_grounded": id_i in self.grounded,
                        f"{id_j}_grounded": id_j in self.grounded,
                        f"{id_i}_cas": round(float(self.bs.traf.cas[i]), 1),
                        f"{id_j}_cas": round(float(self.bs.traf.cas[j]), 1),
                        "route": vh_i.current_route_section,
                    }
                )

    def _enforce_holding_spacing(self, alt_targets, d):
        """
        Any two aircraft converging on the same intersection -- regardless
        of route -- need spacing from each other while holding/queued, the
        same way in-trail same-route aircraft do. This is the gap the
        altitude-arrival gate exposed: once aircraft can be held longer
        (waiting to finish climbing before crossing), more of them end up
        queued near the same intersection at once, and nothing was
        preventing two queued aircraft on DIFFERENT routes from ending up
        stacked on each other.

        Ordered by live distance to the intersection point (closer =
        ahead), not distflown or route history -- this is the only ordering
        that's meaningful across different routes converging on one point.
        Uses a bigger buffer than same-route spacing and is meant to
        trigger earlier: a halted aircraft can't back away once it's
        already stopped, so it needs to be told to stop before it's close,
        not at the moment it's already close.
        """
        n_ac = self.bs.traf.lat.shape[0]
        if n_ac == 0:
            return

        dist_to_target = {}  # i -> (target_intersection, distance)
        for i in range(n_ac):
            id_ = self.bs.traf.id[i]
            vh = self.vehicle_helpers.get(id_)
            if vh is None:
                continue
            # Only aircraft still APPROACHING an intersection belong here --
            # "closer = ahead in the queue, farther = trailing" only makes
            # sense for convergence. An aircraft already inside an
            # intersection and departing along its route is moving AWAY
            # from that point by design; treating "farther from BC" as
            # "trailing" for a BC-departing aircraft is backwards -- it
            # halts whichever aircraft manages to make real progress, and
            # both end up deadlocked near the intersection boundary. Same-
            # route spacing and round-robin already govern separation once
            # an aircraft is inside/departing, so this mechanism skips them.
            if vh.current_intersection is not None:
                continue
            target = getattr(vh, "next_intersection", None)
            if target is None or target not in self.traffic_manager.intersections:
                continue
            lat_i, lon_i = self.traffic_manager.intersections[target].location
            dist = geo.kwikdist(self.bs.traf.lat[i], self.bs.traf.lon[i], lat_i, lon_i) * geo.nm
            dist_to_target[i] = (target, dist)

        for i, (target_i, dist_i) in dist_to_target.items():
            id_i = self.bs.traf.id[i]
            for j, (target_j, dist_j) in dist_to_target.items():
                if i == j or target_i != target_j:
                    continue
                # Only the aircraft farther from the intersection (trailing
                # in the holding queue) halts.
                if dist_j >= dist_i:
                    continue
                if d[i, j] < self.holding_spacing:
                    if id_i not in self.action_override:
                        self.action_override.append(id_i)
                    self.holding_halted_ids.add(id_i)
                    # Don't freeze altitude if this aircraft already has a
                    # granted slot it's climbing toward -- it should keep
                    # climbing while it holds position.
                    if id_i not in alt_targets and id_i not in self.alt_override:
                        self.alt_override.append(id_i)
                    break

    def _hold_for_takeoff(self, d):
        """
        A newly-spawned aircraft that would appear within LOS distance of
        already-active traffic shouldn't be treated as "taking off into a
        conflict" -- it just hasn't taken off yet. Mirrors the original
        runner's within_LOS spawn-clearance check (deny clearance / hold on
        ground if within 1.5x LOS of existing traffic), scoped to the
        takeoff moment only: once released, an aircraft stays released --
        this isn't a general ongoing conflict check, that's what the
        heuristics and spacing methods are for.

        Uses LATERAL distance only, ignoring altitude. Two aircraft on the
        same route section converge to the same shared altitude level
        shortly after spawn (round-robin assigns one level per route
        section, not per aircraft) -- a transient altitude difference at
        the exact moment of spawn (one still climbing from the ground,
        timed slightly differently) isn't a durable signal that they're
        actually clear of each other. Checking 3D distance let two
        aircraft that were laterally right on top of each other but
        briefly at different altitudes both pass the clearance check, only
        to end up in LOS once they leveled off at the same altitude.
        """
        n_ac = self.bs.traf.lat.shape[0]
        if n_ac == 0:
            return

        for i in range(n_ac):
            id_i = self.bs.traf.id[i]
            if id_i in self.released:
                continue  # already cleared once, permanently -- not re-checked

            conflict = False
            for j in range(n_ac):
                if i == j:
                    continue
                id_j = self.bs.traf.id[j]
                if d[i, j] < 1.5 * self.LOS:
                    conflict = True
                    break

            if conflict:
                self.grounded.add(id_i)
            else:
                self.grounded.discard(id_i)
                self.released.add(id_i)

        for id_ in self.grounded:
            if id_ not in self.action_override:
                self.action_override.append(id_)
            if id_ not in self.alt_override:
                self.alt_override.append(id_)

    def _gate_on_altitude_arrival(self, alt_targets):
        """
        An aircraft that's been granted a slot must actually reach its
        assigned altitude before it crosses -- but it doesn't need to stop
        moving to do that. It can keep flying in while it climbs/descends;
        the only hard requirement is that it actually be at the target
        altitude before it gets within LOS of the intersection boundary
        itself. Only once it's that close AND still not at altitude does
        it actually need to stop and finish climbing before proceeding --
        stopping earlier than that has no safety benefit, it just adds
        unnecessary halting time.

        Scoped to this gate only -- same-route and holding spacing halts
        are untouched by this, since those exist because a specific
        nearby aircraft is close right now, not because of a distance-to-
        boundary threshold. Applies regardless of which heuristic assigned
        the target.
        """
        for i in range(self.bs.traf.lat.shape[0]):
            id_ = self.bs.traf.id[i]
            if id_ not in alt_targets:
                continue
            target = alt_targets[id_]
            current_alt_ft = self.meters_to_feet(self.bs.traf.alt[i])
            if abs(current_alt_ft - target) <= self.alt_arrival_tolerance:
                continue  # already at (or close enough to) its assigned altitude

            vh = self.vehicle_helpers.get(id_)
            if vh is None:
                continue
            # Use the SAME identity round-robin's own grouping relies on
            # (self.last_free_route_section), not a separately-computed
            # current_intersection/next_intersection -- those two can
            # disagree about which intersection governs this aircraft
            # right now, and when they do, this gate can end up measuring
            # distance to the wrong point entirely, concluding "still far
            # away" right as the aircraft is actually about to cross the
            # real boundary without having finished its climb.
            route_section = self.last_free_route_section.get(id_, vh.current_route_section)
            if route_section is None or len(route_section) != 4:
                continue
            target_intersection = route_section[2:4]
            if target_intersection not in self.traffic_manager.intersections:
                continue
            lat_t, lon_t = self.traffic_manager.intersections[target_intersection].location
            dist_to_center = geo.kwikdist(self.bs.traf.lat[i], self.bs.traf.lon[i], lat_t, lon_t) * geo.nm
            halt_threshold = self._get_intersection_radius_m(target_intersection) + self.LOS + self._stopping_distance_margin(i)
            if dist_to_center > halt_threshold:
                continue  # still room to close in while climbing/descending

            if id_ not in self.action_override:
                self.action_override.append(id_)
            # Deliberately NOT added to alt_override -- it must keep
            # climbing/descending toward target, just not move forward
            # while doing so.

    def reset(self, scenario_file_override=None):
        """
        Start a new episode. Returns nothing; call step() in a loop after
        this. scenario_file_override, if given, is used exactly as-is
        (must be a specific .scn path) instead of the usual random draw
        from self.scen_file's directory -- lets a caller deterministically
        sweep every scenario in a set exactly once, for a paired
        comparison against another method run the same way.
        """
        self.step_counter = 0
        self.ever_had_aircraft = False  # distinguishes "no aircraft YET" (too
                                         # early, more are scheduled to spawn)
                                         # from "no aircraft LEFT" (genuinely done)
        self.episode_done = False
        self.los_counter = 0
        self.nmacs = 0        # matches D2MAV_A/runner.py's convention exactly (see _flag_nmacs)
        self.nmac_time = 0
        self.acInfo = {}      # ac_id -> {"NMAC": [0/1 per step since spawn]}, same structure they use
        self.max_noise_increase = 0
        self.average_noise_increase = {}  # route_id/intersection_id -> list of noise_increase samples
        self.los_events = 0
        self.los_same_route_count = 0
        self.los_diff_route_count = 0
        self.last_alt_targets = {}
        self.grounded = set()   # ac_ids held on the ground, not yet released to fly
        self.released = set()   # ac_ids that have been cleared for takeoff (once, permanent)
        self.pair_history = {}  # sorted (id_i, id_j) -> deque of per-step snapshots,
                                 # dumped in full the moment that pair gets a NEW LOS
        self.stop_after_los = False  # set True once the first LOS event fires (debug mode only)
        self.same_route_halted_ids = set()
        self.holding_halted_ids = set()
        self.last_free_route_section = {}  # ac_id -> route_section, updated every step
                                            # (always live now; kept for diagnostics)
        self.last_assigned_alt = {}  # ac_id -> most recent altitude actually assigned,
                                      # by any system -- used to hold altitude steady
                                      # while an aircraft is physically within an
                                      # intersection, regardless of which round-robin
                                      # system (departing or arriving) currently governs it
        self.alt_adjustment_events = []  # flat list of individual climb events,
                                          # each value = levels traversed by that
                                          # one climb (descents excluded) -- pool
                                          # and aggregate however is appropriate
                                          # at reporting time, not baked in here
        self.full_alt_adjustments = {}  # ac_id -> [levels traversed, per climb event]
        self.last_commanded_alt_target = {}  # ac_id -> last genuinely-commanded target,
                                              # used to detect climbs above
        self.nmac_event_lengths = []  # flat list of individual NMAC streak durations
                                       # (seconds), raw per-event -- see _check_goals_and_los
        self.prev_LOS_pairs = []
        self.full_travel = {}
        self.travel_start = {}
        self.full_halting_times = {}
        self.halt_start = {}
        self.midair_halt_start = {}
        self.midair_halting_times = []  # like halting_times, but excludes ground/takeoff
                                          # holds entirely -- only counts halt time for
                                          # aircraft already released to fly (round-robin
                                          # queue, same-route spacing, altitude gate).
                                          # RL's policy has no speed-control action at all
                                          # (self.speed_dim = 0 in D2MAV_A/runner.py), so
                                          # this is directly comparable to a flat 0 there.
        self.halting_times = []
        self.airborne_count_history = []  # per-step count of aircraft in the
                                           # sim not currently held for takeoff
        self.first_halt_record = {}  # ac_id -> (step, approach_dist, target_intersection)
                                      # at the moment it was FIRST halted while
                                      # approaching, ungranted -- lets us check
                                      # later whether it was stopped in time
                                      # before physically entering the zone

        # Heuristic-shared state
        self.action_override = []  # ac_ids to halt (speed = 0) this step
        self.alt_override = []     # ac_ids to hold current altitude this step

        # CSMA/CD state
        self.wait_time = {}        # ac_id -> remaining random-backoff steps

        # Round-robin state: per intersection, a FIFO queue of route
        # sections waiting for a slot, and a dict of currently-active
        # {route_section: level}. Slots are keyed by route section, not by
        # aircraft -- any number of aircraft on the same route section share
        # one slot, since they don't cross each other; only different route
        # sections converging on the same intersection compete for levels.
        self.rr_queue = {}
        self.rr_active = {}
        self.rr_granted_step = {}  # intersection_id -> {route_section: step granted} --
                                    # used both for rotation timing and long-held-slot diagnostics
        self.rr_draining = {}  # intersection_id -> {route_section: frozenset(ids allowed
                                # to finish crossing)} -- present only while that
                                # route_section's slot is being rotated out
        self.last_rr_groups = {}  # (intersection_id, route_section) -> [ac_id, ...], for diagnostics

        if scenario_file_override is not None:
            scenario_file = scenario_file_override
        elif ".scn" not in self.scen_file:
            scenario_files = glob.glob(f"{self.scen_file}" + "/*.scn")
            scenario_file = np.random.choice(scenario_files, 1)[0]
        else:
            scenario_file = self.scen_file
        self.scen_file_temp = scenario_file

        self.create_traffic_manager()

        ic_path = self.working_directory + "/" + scenario_file
        if self.debug:
            print(f"[reset] IC path: {ic_path}")
            print(f"[reset] IC path exists on disk: {os.path.isfile(ic_path)}")

        self.bs.stack.stack("IC " + ic_path)
        self.bs.stack.stack("FF")
        self.bs.sim.step()
        self.bs.stack.stack("FF")

        if self.debug:
            print(f"[reset] aircraft in traf after first FF+step: {len(self.bs.traf.id)} -- {list(self.bs.traf.id)[:10]}")

        before = self.bs.sim.simt
        self.bs.sim.step()
        after = self.bs.sim.simt
        if (after - before) == 0:
            if self.debug:
                print(f"[reset] sim time did not advance ({before} -> {after}) -- IC likely failed to load at all, retrying")
            return self.reset(scenario_file_override=scenario_file_override)
        assert (after - before) == self.simdt
        self.step_counter += 1

        if self.debug:
            print(f"[reset] aircraft in traf after second step (sim time DID advance): {len(self.bs.traf.id)} -- {list(self.bs.traf.id)[:10]}")
            if len(self.bs.traf.id) == 0:
                print(f"[reset] zero aircraft this early is expected if the scenario's first CRE is scheduled more than ~2-3s in -- not an error")

        for i in range(self.bs.traf.lat.shape[0]):
            id_ = self.bs.traf.id[i]
            if id_ not in self.vehicle_helpers.keys():
                route_name = self.bs.traf.ap.route[i].wpname[0][0:-1]
                self.vehicle_helpers[id_] = VehicleHelper(id_, self.routes_loaded[route_name])
            vh = self.vehicle_helpers[id_]
            if vh.next_intersection is None:
                # qatc.py's VehicleHelper never computes this itself (it's a
                # documented TODO there) -- D2MAV_A/runner.py sets it this
                # same way. Without it, round-robin has nothing to queue
                # against until an aircraft is already inside an
                # intersection, which is too late to prevent LOS.
                vh.next_intersection = vh.route.route_id[2:4]
            self.travel_start[id_] = self.bs.sim.simt
            self.full_halting_times[id_] = []
            self.wait_time[id_] = 0
            self.full_alt_adjustments[id_] = []

        if self.gui:
            self.bs.net.update()

    def _update_route_progress(self):
        """
        Determine each aircraft's current and next route section directly
        from its BlueSky flight plan (the autopilot's active waypoint list),
        exactly matching the logic in D2MAV_A/runner.py's step(). This is
        deterministic -- it reads the actual route the aircraft is flying --
        so next_intersection stays correct across multi-hop routes instead
        of going stale after the first hop (which was the bug: a value
        cached once at spawn, or a live geometric nearest-intersection
        guess, both under- and over-shot what the flight plan already
        states outright).
        """
        for i in range(self.bs.traf.lat.shape[0]):
            id_ = self.bs.traf.id[i]
            vh = self.vehicle_helpers.get(id_)
            if vh is None:
                continue

            autopilot_route = self.bs.traf.ap.route[i]
            route_counter = 0
            current_route_section = None
            next_route_section = None

            found = False
            while not found and autopilot_route.iactwp + route_counter < len(autopilot_route.wpname):
                wp_string = autopilot_route.wpname[autopilot_route.iactwp + route_counter]
                section = wp_string[0:4]
                if section in self.routes_loaded:
                    current_route_section = section
                    found = True
                else:
                    route_counter += 1

            found = False
            while not found and autopilot_route.iactwp + route_counter < len(autopilot_route.wpname):
                wp_string = autopilot_route.wpname[autopilot_route.iactwp + route_counter]
                section = wp_string[0:4]
                if section in self.routes_loaded and section != current_route_section:
                    next_route_section = section
                    found = True
                else:
                    route_counter += 1

            vh.current_route_section = current_route_section
            vh.next_route_section = next_route_section  # the segment this aircraft
                                                          # will be on after its current
                                                          # one -- lets spacing checks see
                                                          # it's about to merge onto a
                                                          # segment before it actually does
            if current_route_section is not None:
                if current_route_section in self.routes_loaded:
                    vh.route = self.routes_loaded[current_route_section]
                vh.next_intersection = current_route_section[2:4]
            elif next_route_section is not None:
                vh.next_intersection = next_route_section[2:4]

    def _update_intersection_tracking(self):
        """
        Determine which aircraft are currently inside which intersections,
        and register newly-spawned aircraft with a VehicleHelper. Lifted
        from D2MAV_A/runner.py's step() -- that's the part of the
        traffic-manager machinery still "live" in this repo (used there
        purely for intruder-detection bookkeeping; here it doubles as our
        conflict/intersection-entry detector for both heuristics).
        """
        for i in range(self.bs.traf.lat.shape[0]):
            id_ = self.bs.traf.id[i]
            curr_gps = [self.bs.traf.lon[i], self.bs.traf.lat[i]]

            if id_ not in self.vehicle_helpers.keys():
                route_name = self.bs.traf.ap.route[i].wpname[0][0:-1]
                self.vehicle_helpers[id_] = VehicleHelper(id_, self.routes_loaded[route_name])
                self.travel_start[id_] = self.bs.sim.simt
                self.full_halting_times[id_] = []
                self.wait_time[id_] = 0
                self.full_alt_adjustments[id_] = []

            vh = self.vehicle_helpers[id_]
            if vh.next_intersection is None:
                vh.next_intersection = vh.route.route_id[2:4]
            found_intersection = None
            for intersection in self.traffic_manager.intersections.values():
                if self.traffic_manager.check_if_within_intersection(curr_gps, intersection.tower_ID):
                    found_intersection = intersection.tower_ID
                    break

            if found_intersection is not None:
                if not vh.within_intersection:
                    vh.within_intersection = True
                    vh.current_intersection = found_intersection
                    if self.debug and id_ in self.first_halt_record and id_ in self.action_override:
                        halt_step, halt_dist, halt_target = self.first_halt_record[id_]
                        elapsed_steps = self.step_counter - halt_step
                        print(
                            f"[step {self.step_counter}] {id_} entered {found_intersection} "
                            f"while STILL halted/ungranted -- was halted at step {halt_step} "
                            f"(dist={halt_dist:.1f}m from {halt_target}), {elapsed_steps} steps ago"
                        )
            elif vh.within_intersection:
                vh.within_intersection = False
                vh.current_intersection = None

    # ------------------------------------------------------------------
    # Heuristic 1: round-robin altitude assignment
    # ------------------------------------------------------------------
    def _round_robin_actions(self):
        n_ac = self.bs.traf.lat.shape[0]

        # Group live aircraft by (target intersection, route section).
        # Aircraft sharing a route section through an intersection are
        # entering and exiting at the same points -- they don't cross each
        # other, so they share ONE altitude slot with no cap on how many of
        # them use it. Only DIFFERENT route sections converging on the same
        # intersection actually compete for the limited number of altitude
        # levels. (In-trail spacing between same-route aircraft is handled
        # separately by _enforce_same_route_spacing(), called from step().)
        #
        # target and route_section are always derived from the aircraft's
        # LIVE current_route_section (see the loop below for why this
        # replaced an earlier within_intersection-based lock).
        groups = {}  # (intersection_id, route_section) -> [ac_id, ...]
        approach_dist = {}  # ac_id -> distance to target intersection, only for
                             # aircraft still approaching (not yet inside) --
                             # used to decide when a halt actually takes effect,
                             # separately from when it joins the queue
        for i in range(n_ac):
            id_ = self.bs.traf.id[i]
            vh = self.vehicle_helpers.get(id_)
            if vh is None or vh.current_route_section is None:
                continue

            if vh.within_intersection:
                # Hold the identity from the moment this aircraft entered
                # the intersection for the WHOLE passage -- inbound
                # approach, physical transit, and outbound departure --
                # until it's genuinely clear (within_intersection becomes
                # False). One clearance for the whole crossing, not a
                # second one partway through just because the flight
                # plan's route label happened to flip. Simpler, and
                # accepted for this baseline: an intersection with a large
                # configured radius can delay this aircraft's registration
                # at the NEXT intersection for a while, since it's still
                # "in" this one by that geometric definition until it
                # clears -- that's a known, accepted cost, not an oversight.
                route_section = self.last_free_route_section.get(id_, vh.current_route_section)
            else:
                route_section = vh.current_route_section
                self.last_free_route_section[id_] = route_section

            if len(route_section) != 4:
                continue

            # Genuine takeoff: not yet released to fly. self.grounded is a
            # direct behavioral flag (set/cleared by _hold_for_takeoff),
            # not a geometric proxy.
            if id_ in self.grounded:
                continue

            target = route_section[2:4]
            if target not in self.traffic_manager.intersections:
                continue

            already_within_target = vh.within_intersection and vh.current_intersection == target
            if not already_within_target:
                lat_i, lon_i = self.traffic_manager.intersections[target].location
                dist = geo.kwikdist(self.bs.traf.lat[i], self.bs.traf.lon[i], lat_i, lon_i) * geo.nm
                if dist > self.queue_radius:
                    continue  # not near it yet -- don't reserve a slot early
                approach_dist[id_] = dist
                # No approach_dist entry when already_within_target is True
                # (the aircraft is already there) -- that absence is what
                # the halt-decision step below uses to guarantee it's never
                # forcibly halted while physically inside an intersection.

            groups.setdefault((target, route_section), []).append(id_)

        # Free slots whose route section has no live traffic left near/in
        # the intersection. Also apply real rotation, in two phases, so a
        # popular route_section can't starve everyone else converging on
        # the same intersection forever:
        #
        #   1. Once a route_section has held its slot past
        #      max_slot_hold_steps, with something else actually waiting
        #      behind it, it starts DRAINING: the aircraft already
        #      assigned to it (snapshotted right now) get to finish
        #      crossing normally, but no NEW aircraft on that same
        #      route_section are admitted to the slot while it drains.
        #   2. Once every snapshotted aircraft has actually left (not
        #      before -- never evict mid-crossing, that would strand an
        #      aircraft without permission), the slot is freed for real,
        #      the timer resets, and the route_section naturally re-enters
        #      the queue on equal footing with whatever's waiting -- any
        #      aircraft that arrived during the drain and got blocked
        #      simply becomes part of that fresh cycle.
        for intersection_id, active in self.rr_active.items():
            granted = self.rr_granted_step.get(intersection_id, {})
            queue = self.rr_queue.get(intersection_id, deque())
            draining = self.rr_draining.setdefault(intersection_id, {})
            for route_section in list(active.keys()):
                key = (intersection_id, route_section)
                live_members = groups.get(key, [])

                if route_section in draining:
                    still_finishing = [id_ for id_ in live_members if id_ in draining[route_section]]
                    if not still_finishing:
                        del active[route_section]
                        granted.pop(route_section, None)
                        draining.pop(route_section, None)
                    continue

                if not live_members:
                    del active[route_section]
                    granted.pop(route_section, None)
                    continue

                held_for = self.step_counter - granted.get(route_section, self.step_counter)
                if held_for >= self.max_slot_hold_steps and queue:
                    draining[route_section] = frozenset(live_members)

        # Queue any route section with live traffic that isn't active yet.
        for (intersection_id, route_section), ids in groups.items():
            active = self.rr_active.setdefault(intersection_id, {})
            queue = self.rr_queue.setdefault(intersection_id, deque())
            if route_section in active or route_section in queue:
                continue
            queue.append(route_section)

        # Promote queued route sections into free slots, FIFO (round-robin).
        for intersection_id, queue in self.rr_queue.items():
            active = self.rr_active.setdefault(intersection_id, {})
            granted = self.rr_granted_step.setdefault(intersection_id, {})
            used_levels = set(active.values())
            free_levels = [lvl for lvl in self.alt_levels if lvl not in used_levels]
            while queue and free_levels:
                route_section = queue[0]
                if (intersection_id, route_section) not in groups:
                    queue.popleft()  # cleared out while waiting
                    continue
                active[route_section] = free_levels.pop(0)
                granted[route_section] = self.step_counter
                queue.popleft()

        # Translate groups into per-aircraft halt / altitude-target decisions.
        halted_ids = set()
        halted_intersection = {}  # id_ -> intersection_id, for queued aircraft
        alt_targets = {}
        for (intersection_id, route_section), ids in groups.items():
            active = self.rr_active.get(intersection_id, {})
            draining_snapshot = self.rr_draining.get(intersection_id, {}).get(route_section)
            # An aircraft without a granted slot doesn't need to physically
            # stop the moment it's registered in the queue (queue_radius is
            # 8100m -- far too early to actually halt). It keeps cruising
            # in and only actually stops once continuing would put it
            # within LOS of the intersection boundary itself, plus a
            # stopping-distance margin (see _stopping_distance_margin) so
            # there's actually enough room to decelerate in time.
            base_halt_threshold = self._get_intersection_radius_m(intersection_id) + self.LOS
            if route_section in active:
                level = active[route_section]
                for id_ in ids:
                    if draining_snapshot is None or id_ in draining_snapshot:
                        alt_targets[id_] = level
                        continue
                    # Arrived after this route_section's slot started
                    # draining -- not part of the finishing batch, so it
                    # waits for the next cycle instead of joining crossing
                    # traffic that's already being phased out.
                    vh_ = self.vehicle_helpers.get(id_)
                    if vh_ is not None and vh_.within_intersection:
                        continue  # never halt physically inside an intersection
                    dist = approach_dist.get(id_)
                    idx = self.bs.traf.id2idx(id_)
                    halt_threshold = base_halt_threshold + (self._stopping_distance_margin(idx) if idx != -1 else 0)
                    if dist is not None and dist > halt_threshold:
                        continue  # still far out -- keep cruising, not time to stop yet
                    halted_ids.add(id_)
                    halted_intersection[id_] = intersection_id
                    if id_ not in self.first_halt_record:
                        self.first_halt_record[id_] = (self.step_counter, dist, intersection_id)
            else:
                for id_ in ids:
                    vh_ = self.vehicle_helpers.get(id_)
                    if vh_ is not None and vh_.within_intersection:
                        continue  # never halt physically inside an intersection --
                                  # not yet granted, but it keeps moving at its
                                  # last assigned altitude (see the hold below)
                                  # until it gets a fresh grant or clears the zone
                    dist = approach_dist.get(id_)
                    idx = self.bs.traf.id2idx(id_)
                    halt_threshold = base_halt_threshold + (self._stopping_distance_margin(idx) if idx != -1 else 0)
                    if dist is not None and dist > halt_threshold:
                        continue  # still far out -- keep cruising, not time to stop yet
                    halted_ids.add(id_)
                    halted_intersection[id_] = intersection_id
                    if id_ not in self.first_halt_record:
                        self.first_halt_record[id_] = (self.step_counter, dist, intersection_id)

        self.action_override = list(halted_ids)
        self.rr_halted_ids = set(halted_ids)  # for pair-history halt-reason diagnostics

        # Queued aircraft hold position while they wait -- but they
        # shouldn't hold at an altitude that happens to coincide with an
        # already-active route_section's granted level at the same
        # intersection. That's a real conflict (an aircraft with no
        # assigned level at all sitting at the exact altitude another
        # route_section is actively using), not a harmless coincidence:
        # blindly holding "wherever it happened to be when queued" doesn't
        # check against traffic that's actually active there. Give it an
        # escape target to a currently-unused level instead; it still
        # doesn't move forward (stays in action_override), just clear of
        # the level in use.
        alt_override_ids = []
        for id_ in halted_ids:
            idx = self.bs.traf.id2idx(id_)
            if idx == -1:
                continue
            active = self.rr_active.get(halted_intersection.get(id_), {})
            used_levels = set(active.values())
            current_alt_ft = self.meters_to_feet(self.bs.traf.alt[idx])
            nearest_used = min(used_levels, key=lambda lvl: abs(lvl - current_alt_ft)) if used_levels else None
            if nearest_used is not None and abs(current_alt_ft - nearest_used) < self.alt_level_separation / 2:
                free_levels = [lvl for lvl in self.alt_levels if lvl not in used_levels]
                if free_levels:
                    alt_targets[id_] = min(free_levels, key=lambda lvl: abs(lvl - current_alt_ft))
                    continue  # don't hold-in-place -- let it move to the escape level
            alt_override_ids.append(id_)

        # An aircraft physically within an intersection must hold its
        # assigned altitude until it's genuinely clear -- not switch early
        # to whatever a DOWNSTREAM intersection's round-robin system would
        # assign once its route_section flips mid-crossing. Without this,
        # two aircraft departing the same hub toward different
        # destinations get managed by two independent, uncoordinated
        # systems while still physically close together right where their
        # paths just diverged -- the different-route LOS pattern this was
        # built to fix. Purely an altitude hold, not a halt: the aircraft
        # keeps moving, it just isn't handed off to the next system's
        # commands until region_shape says it's actually clear.
        for i in range(n_ac):
            id_ = self.bs.traf.id[i]
            vh = self.vehicle_helpers.get(id_)
            if vh is None:
                continue
            if vh.within_intersection and id_ not in alt_targets:
                current_intersection = vh.current_intersection
                used_levels_here = set(self.rr_active.get(current_intersection, {}).values()) if current_intersection else set()
                fallback = self.last_assigned_alt.get(id_)
                current_alt_ft = round(self.meters_to_feet(self.bs.traf.alt[i]))

                if fallback is not None and fallback not in used_levels_here:
                    alt_targets[id_] = fallback
                else:
                    # Either no prior assignment to fall back to, or the
                    # fallback collides with a level ACTIVELY granted at
                    # the intersection this aircraft is physically inside
                    # right now (a stale value left over from an earlier,
                    # different crossing -- reusing it blindly is exactly
                    # what let two aircraft on different route_sections
                    # end up sharing an altitude while still close
                    # together). Pick a genuinely free level here instead,
                    # same approach as the existing escape-altitude logic
                    # for queued aircraft.
                    free_levels = [lvl for lvl in self.alt_levels if lvl not in used_levels_here]
                    if free_levels:
                        alt_targets[id_] = min(free_levels, key=lambda lvl: abs(lvl - current_alt_ft))
                    elif fallback is not None:
                        alt_targets[id_] = fallback  # no free level at all -- imperfect, but still better than unmanaged
                    else:
                        alt_targets[id_] = current_alt_ft
            if id_ in alt_targets:
                self.last_assigned_alt[id_] = alt_targets[id_]

        self.alt_override = alt_override_ids
        self.last_rr_groups = groups  # for diagnostics -- lets the debug print check
                                       # the exact same "still actually blocking this
                                       # slot" membership the real logic uses, instead
                                       # of a looser "still exists in the sim" proxy
        return alt_targets

    # ------------------------------------------------------------------
    # Heuristic 2: decentralized CSMA/CD-style altitude negotiation
    # ------------------------------------------------------------------
    def _csma_cd_actions(self, geometries, d):
        n_ac = self.bs.traf.lat.shape[0]
        self.action_override = []
        self.alt_override = []
        alt_targets = {}

        for i in range(n_ac):
            id_ = self.bs.traf.id[i]
            if id_ not in self.wait_time:
                self.wait_time[id_] = 0

            own_level = min(self.alt_levels, key=lambda lvl: abs(lvl - self.meters_to_feet(self.bs.traf.alt[i])))

            nearby_levels = set()
            in_conflict = False
            for j in range(n_ac):
                if i == j or not self.bs.traf.active[j]:
                    continue
                if d[i, j] > self.intruderThreshold:
                    continue
                other_level = min(
                    self.alt_levels, key=lambda lvl: abs(lvl - self.meters_to_feet(self.bs.traf.alt[j]))
                )
                nearby_levels.add(other_level)
                if other_level == own_level and geometries.geoms[i].intersects(geometries.geoms[j]):
                    in_conflict = True

            # Currently backing off from a previous conflict: keep halted.
            if self.wait_time[id_] > 0:
                self.wait_time[id_] -= 1
                self.action_override.append(id_)
                self.alt_override.append(id_)
                continue

            # New conflict: start a random backoff before re-checking.
            if in_conflict:
                self.wait_time[id_] = random.randint(self.backoff_min_steps, self.backoff_max_steps)
                self.action_override.append(id_)
                self.alt_override.append(id_)
                continue

            # No conflict / backoff just expired: look for a free level
            # among nearby traffic and move to the closest one if available.
            free_levels = [lvl for lvl in self.alt_levels if lvl not in nearby_levels]
            if free_levels:
                alt_targets[id_] = min(free_levels, key=lambda lvl: abs(lvl - own_level))
            # else: no free level nearby -- fall through and hold current
            # altitude; conflict (if any persists) will be re-detected and
            # trigger a fresh backoff next step.

        return alt_targets

    # ------------------------------------------------------------------
    # Simulation step
    # ------------------------------------------------------------------
    def step(self):
        """Advances the simulation by one step using the selected heuristic."""
        self.same_route_halted_ids = set()  # recomputed fresh below, for diagnostics
        self.holding_halted_ids = set()     # recomputed fresh below, for diagnostics

        # Aircraft that exist in the sim and aren't currently held for
        # takeoff -- i.e. actually in the air right now. Matches the RL
        # side's airborne_count_history for a direct comparison, using the
        # more precise signal available here (self.grounded) rather than a
        # travel_start-based proxy.
        self.airborne_count_history.append(
            sum(1 for id_ in self.bs.traf.id if id_ not in self.grounded)
        )

        coord_transform = self.transformer.transform(self.bs.traf.lon, self.bs.traf.lat)
        geometries = MultiLineString(
            [
                [(coord_transform[0][i], coord_transform[1][i])]
                + [
                    self.transformer.transform(
                        self.bs.traf.ap.route[i].wplon[j], self.bs.traf.ap.route[i].wplat[j]
                    )
                    for j in range(
                        self.bs.traf.ap.route[i].iactwp, len(self.bs.traf.ap.route[i].wplon)
                    )
                ]
                for i in range(self.bs.traf.lat.shape[0])
            ]
        )

        self._update_route_progress()
        self._update_intersection_tracking()

        d = self._pairwise_distance_matrix()

        if self.heuristic == "round_robin":
            alt_targets = self._round_robin_actions()
        else:
            alt_targets = self._csma_cd_actions(geometries, d)
        self.last_alt_targets = alt_targets  # for the LOS diagnostic below

        # A newly-spawned aircraft that would appear on top of existing
        # traffic hasn't "taken off into a conflict" -- it just doesn't
        # take off yet.
        self._hold_for_takeoff(d)

        # Must finish climbing/descending to the assigned level before
        # being cleared to cross -- applies regardless of which heuristic
        # assigned the target.
        self._gate_on_altitude_arrival(alt_targets)

        # Applies regardless of which heuristic ran: neither one manages
        # same-route in-trail spacing on its own.
        self._enforce_same_route_spacing(d)

        # Applies regardless of route: aircraft holding/queued near the
        # same intersection need spacing from each other too.
        self._enforce_holding_spacing(alt_targets, d)

        # Recorded here, after every halt-deciding method has run, so
        # "halted" reflects the actual final decision, not a mid-pipeline
        # snapshot. Debug-only: this is pure diagnostic bookkeeping with
        # no effect on simulation behavior, not worth the O(n^2) pass
        # every step when nobody's looking at it.
        if self.debug:
            self._record_pair_history(d)

        for i in range(self.bs.traf.lat.shape[0]):
            id_ = self.bs.traf.id[i]

            if id_ in self.action_override:
                speed = 0
                if id_ not in self.halt_start:
                    self.halt_start[id_] = self.bs.sim.simt
            else:
                speed = self.speeds[2]
                if id_ in self.halt_start:
                    halting_time = self.bs.sim.simt - self.halt_start[id_]
                    del self.halt_start[id_]
                    self.halting_times.append(halting_time)
                    self.full_halting_times.setdefault(id_, []).append(halting_time)

            # Midair-only: same start/stop logic, but only while the
            # aircraft has already been released to fly (not grounded
            # awaiting takeoff clearance) -- isolates hovering-in-place
            # time from ground-hold time, which the conflated metric above
            # doesn't distinguish.
            if id_ in self.action_override and id_ not in self.grounded:
                if id_ not in self.midair_halt_start:
                    self.midair_halt_start[id_] = self.bs.sim.simt
            else:
                if id_ in self.midair_halt_start:
                    midair_halt_time = self.bs.sim.simt - self.midair_halt_start[id_]
                    del self.midair_halt_start[id_]
                    self.midair_halting_times.append(midair_halt_time)

            self.bs.stack.stack("{} SPD {}".format(id_, speed))

            current_alt_ft = round(self.meters_to_feet(self.bs.traf.alt[i]))
            if id_ in self.alt_override:
                target_alt = current_alt_ft
            elif id_ in alt_targets:
                target_alt = alt_targets[id_]
                last_target = self.last_commanded_alt_target.get(id_)
                if last_target is not None and target_alt > last_target:
                    # Only climbs count -- descents aren't charged as an
                    # adjustment cost. Levels traversed, not just "did the
                    # target change": climbing 3 levels in one reassignment
                    # is a bigger cost than climbing 1, so record the
                    # actual count. Each event is its own raw value in a
                    # flat list, not folded into a running per-aircraft
                    # count -- how to aggregate (per-aircraft mean, pooled
                    # mean, etc.) is a reporting decision, not something to
                    # bake in here.
                    levels_climbed = (target_alt - last_target) / self.alt_level_separation
                    self.alt_adjustment_events.append(levels_climbed)
                    self.full_alt_adjustments.setdefault(id_, []).append(levels_climbed)
                self.last_commanded_alt_target[id_] = target_alt
            else:
                target_alt = current_alt_ft
            target_alt = max(self.min_alt, min(self.max_alt, target_alt))
            self.bs.stack.stack("{} ALT {}".format(id_, target_alt))

        self.bs.sim.step()
        self.step_counter += 1
        if self.gui:
            self.bs.net.update()
            if self.gui_realtime:
                time.sleep(self.simdt / self.gui_speed)

        self._check_goals_and_los()
        self._record_noise()

        n_ac_remaining = self.bs.traf.lat.shape[0]
        if n_ac_remaining > 0:
            self.ever_had_aircraft = True
        done = (
            (self.step_counter > 1 and n_ac_remaining == 0 and self.ever_had_aircraft)
            or self.step_counter >= self.max_steps
            or self.stop_after_los
        )
        return done

    def _record_noise(self):
        """
        Ported directly from D2MAV_A/runner.py's noise-tracking block (same
        coefficients, same dB->power->dB summation, same ambient-subtraction)
        so noise impact is measured identically for the RL policy and this
        heuristic -- an apples-to-apples comparison depends on this being
        the exact same formula, not a similar one.
        """
        noise_vals = {}
        for id_val in self.bs.traf.id:
            idx_val = self.bs.traf.id2idx(id_val)
            ac_alt = self.meters_to_feet(self.bs.traf.alt[idx_val])
            if ac_alt <= self.min_alt:
                ac_noise = 0
            else:
                ac_noise = self.a_0 + (self.a_1 * math.log10(ac_alt)) + (self.a_2 * ((math.log10(ac_alt)) ** 2))
            vh = self.vehicle_helpers.get(id_val)
            if vh is None:
                continue
            if vh.current_route_section is not None:
                noise_vals.setdefault(vh.current_route_section, []).append(10 ** (ac_noise / 10))
            if vh.current_intersection is not None:
                noise_vals.setdefault(vh.current_intersection, []).append(10 ** (ac_noise / 10))

        for route_id in noise_vals.keys():
            if np.sum(noise_vals[route_id]) == 0:
                noise_increase = 0
            else:
                total_noise_impact = 10 * math.log10(np.sum(noise_vals[route_id]))
                ambient_noise_val = self.ambient_noise_level.get(route_id, 40)
                noise_increase = total_noise_impact - ambient_noise_val
            if noise_increase < 0:
                noise_increase = 0
            if noise_increase != 0:
                self.average_noise_increase.setdefault(route_id, []).append(noise_increase)
            if noise_increase >= self.max_noise_increase:
                self.max_noise_increase = noise_increase

    def _flag_nmacs(self, d, geometries):
        """
        Matches D2MAV_A/runner.py's NMAC-flagging logic exactly (not just
        the aggregation -- the actual definition of what counts), for a
        directly comparable number against the RL policy's reported nmacs:

          1. 3D distance (lateral + altitude, meters) < LOS
          2. The two aircraft's remaining flight paths actually intersect
             geometrically -- not just "currently close by coincidence"
          3. Not a reciprocal/swapped route pair (e.g. DTWL vs WLDT)
          4. Not on the same current_route_section
          5. Not matching on route.route_id[0:3] (their secondary same-
             route check, keyed off the route object rather than the
             section string)
          6. MODIFIED from the original: theirs requires both aircraft to
             have nonzero ground speed. In their runner that's a proxy for
             "not still taking off," since aircraft essentially never halt
             mid-flight there. That proxy doesn't hold here -- halting
             mid-flight is a deliberate, frequent mechanic in this
             heuristic, so using gs != 0 verbatim would silently exclude
             most of what the heuristic actually does to stay safe.
             Replaced with the literal intent instead: both aircraft must
             have actually taken off (not in self.grounded), regardless of
             whether they're currently halted for a legitimate heuristic
             reason.

        Appends one 0/1 entry per aircraft per step to self.acInfo, mirroring
        their per-aircraft NMAC history list exactly, so the same groupby-
        based streak counting can be applied on completion.
        """
        n_ac = self.bs.traf.lat.shape[0]
        for i in range(n_ac):
            id_ = self.bs.traf.id[i]
            if id_ not in self.acInfo:
                self.acInfo[id_] = {"NMAC": []}
            self.acInfo[id_]["NMAC"].append(0)  # overwritten below if an NMAC occurred

        for i in range(n_ac):
            id_i = self.bs.traf.id[i]
            vh_i = self.vehicle_helpers.get(id_i)
            for j in range(i + 1, n_ac):
                id_j = self.bs.traf.id[j]
                vh_j = self.vehicle_helpers.get(id_j)

                alt_diff = abs(self.bs.traf.alt[i] - self.bs.traf.alt[j])
                dist_3d = (d[i, j] ** 2 + alt_diff ** 2) ** 0.5
                if dist_3d >= self.LOS:
                    continue

                if not geometries.geoms[i].intersects(geometries.geoms[j]):
                    continue

                rs_i = getattr(vh_i, "current_route_section", None)
                rs_j = getattr(vh_j, "current_route_section", None)
                if rs_i is not None and rs_j is not None and len(rs_i) == 4 and len(rs_j) == 4:
                    if rs_i[2:4] + rs_i[0:2] == rs_j:
                        continue
                    if rs_i == rs_j:
                        continue

                route_i = getattr(vh_i, "route", None)
                route_j = getattr(vh_j, "route", None)
                if route_i is not None and route_j is not None:
                    id_i_prefix = getattr(route_i, "route_id", "")[0:3]
                    id_j_prefix = getattr(route_j, "route_id", "")[0:3]
                    if id_i_prefix == id_j_prefix:
                        continue

                if id_i in self.grounded or id_j in self.grounded:
                    continue

                self.acInfo[id_i]["NMAC"][-1] = 1
                self.acInfo[id_j]["NMAC"][-1] = 1

    def _check_goals_and_los(self):
        n_ac = self.bs.traf.lat.shape[0]
        if n_ac == 0:
            return
        d = self._pairwise_distance_matrix()

        # Rebuilt fresh (post sim.step()), not reused from step()'s pre-step
        # geometries -- aircraft can spawn mid-step, which would otherwise
        # leave this array shorter than self.bs.traf and cause an
        # out-of-range index.
        coord_transform = self.transformer.transform(self.bs.traf.lon, self.bs.traf.lat)
        geometries = MultiLineString(
            [
                [(coord_transform[0][i], coord_transform[1][i])]
                + [
                    self.transformer.transform(
                        self.bs.traf.ap.route[i].wplon[j], self.bs.traf.ap.route[i].wplat[j]
                    )
                    for j in range(
                        self.bs.traf.ap.route[i].iactwp, len(self.bs.traf.ap.route[i].wplon)
                    )
                ]
                for i in range(n_ac)
            ]
        )

        self._flag_nmacs(d, geometries)

        current_los_pairs = []
        for i in range(n_ac):
            for j in range(i + 1, n_ac):
                # bs.traf.alt is in meters, matching d's units (kwikdist * nm).
                alt_diff = abs(self.bs.traf.alt[i] - self.bs.traf.alt[j])
                dist_3d = (d[i, j] ** 2 + alt_diff ** 2) ** 0.5
                if dist_3d < self.LOS:
                    id_i, id_j = self.bs.traf.id[i], self.bs.traf.id[j]

                    # A grounded (not-yet-taken-off) aircraft isn't flying
                    # into a conflict -- it hasn't taken off. Don't count
                    # LOS against it at all while held.
                    if id_i in self.grounded or id_j in self.grounded:
                        continue

                    vh_i = self.vehicle_helpers.get(id_i)
                    vh_j = self.vehicle_helpers.get(id_j)
                    rs_i = getattr(vh_i, "current_route_section", None)
                    rs_j = getattr(vh_j, "current_route_section", None)

                    # Reciprocal (opposite-direction) route sections -- e.g.
                    # "DTWL" and "WLDT" -- run the same two endpoints in
                    # opposite directions. The original runner explicitly
                    # excludes this pairing from intruder detection (it's
                    # not treated as a conflict), so we exclude it from LOS
                    # counting the same way rather than inventing new
                    # separation logic for something the route design
                    # doesn't consider a conflict in the first place.
                    if rs_i is not None and rs_j is not None and len(rs_i) == 4 and len(rs_j) == 4:
                        if rs_i[2:4] + rs_i[0:2] == rs_j:
                            continue

                    pair = tuple(sorted((id_i, id_j)))
                    current_los_pairs.append(pair)
                    if pair not in self.prev_LOS_pairs:
                        self.los_events += 1
                        same_route = rs_i is not None and rs_i == rs_j
                        if same_route:
                            self.los_same_route_count += 1
                        else:
                            self.los_diff_route_count += 1
                        if self.debug:
                            category = "SAME-ROUTE" if same_route else "DIFFERENT-ROUTE"
                            targets = getattr(self, "last_alt_targets", {})
                            alt_i_ft = round(self.meters_to_feet(self.bs.traf.alt[i]))
                            alt_j_ft = round(self.meters_to_feet(self.bs.traf.alt[j]))
                            print(
                                f"[step {self.step_counter}] NEW LOS [{category}]: "
                                f"{id_i} (route={rs_i}, intersection={getattr(vh_i, 'current_intersection', None)}, "
                                f"actual_alt={alt_i_ft}ft, assigned_alt={targets.get(id_i)}ft, halted={id_i in self.action_override}) vs "
                                f"{id_j} (route={rs_j}, intersection={getattr(vh_j, 'current_intersection', None)}, "
                                f"actual_alt={alt_j_ft}ft, assigned_alt={targets.get(id_j)}ft, halted={id_j in self.action_override}) "
                                f"dist_3d={dist_3d:.1f} lateral={d[i, j]:.1f} alt_diff={alt_diff:.1f}"
                            )
                            if same_route:
                                history = self.pair_history.get(pair, [])
                                print(f"[step {self.step_counter}] --- history for {pair} ({len(history)} snapshots) ---")
                                for snap in history:
                                    print(f"    {snap}")

                                # Show the WHOLE chain on this route_section right
                                # now, not just the two aircraft that violated LOS
                                # -- same-route spacing only ever propagates halts
                                # backward down a queue, so the interesting
                                # question is usually who's at the FRONT and why
                                # THEY haven't moved, not the two that happen to
                                # be reported.
                                print(f"[step {self.step_counter}] --- full chain on route {rs_i} ---")
                                target_ni = getattr(vh_i, "next_intersection", None)
                                if target_ni in self.traffic_manager.intersections:
                                    lat_t, lon_t = self.traffic_manager.intersections[target_ni].location
                                    chain = []
                                    for k in range(n_ac):
                                        id_k = self.bs.traf.id[k]
                                        vh_k = self.vehicle_helpers.get(id_k)
                                        if vh_k is None or vh_k.current_route_section != rs_i:
                                            continue
                                        dist_k = geo.kwikdist(self.bs.traf.lat[k], self.bs.traf.lon[k], lat_t, lon_t) * geo.nm
                                        chain.append((dist_k, id_k, k))
                                    chain.sort()  # closest to segment end (= front of queue) first
                                    for dist_k, id_k, k in chain:
                                        vh_k = self.vehicle_helpers.get(id_k)
                                        within = getattr(vh_k, "within_intersection", None)
                                        locked_rs = self.last_free_route_section.get(id_k)
                                        live_rs = getattr(vh_k, "current_route_section", None)
                                        effective_rs = live_rs
                                        effective_target = live_rs[2:4] if live_rs and len(live_rs) == 4 else None
                                        is_grounded = id_k in self.grounded
                                        print(
                                            f"    {id_k}: dist_to_end={dist_k:.1f} "
                                            f"halted={id_k in self.action_override} "
                                            f"reason={self._halt_reason(id_k, k)} "
                                            f"intersection={getattr(vh_k, 'current_intersection', None)} "
                                            f"within_intersection={within} "
                                            f"grounded={is_grounded} "
                                            f"locked_route_section={locked_rs} "
                                            f"EFFECTIVE_target={effective_target} EFFECTIVE_route_section={effective_rs} "
                                            f"assigned_alt={self.last_alt_targets.get(id_k)}"
                                        )

                                    # Is DT (or wherever this chain is headed)
                                    # actually full, or is one of its 5 slots
                                    # occupied by a route_section that's itself
                                    # jammed elsewhere and will never release?
                                    print(f"[step {self.step_counter}] --- active slots at {target_ni} ---")
                                    active = self.rr_active.get(target_ni, {})
                                    granted = self.rr_granted_step.get(target_ni, {})
                                    draining = self.rr_draining.get(target_ni, {})
                                    for route_section, level in active.items():
                                        held_for = self.step_counter - granted.get(route_section, self.step_counter)
                                        if route_section in draining:
                                            snapshot = draining[route_section]
                                            current_group = self.last_rr_groups.get((target_ni, route_section), [])
                                            still_here = [id_ for id_ in snapshot if id_ in current_group]
                                            print(
                                                f"    {route_section}: {level}ft, held {held_for} steps, "
                                                f"DRAINING (snapshot={sorted(snapshot)}, still ACTUALLY blocking={sorted(still_here)})"
                                            )
                                            for id_k in still_here:
                                                k = self.bs.traf.id2idx(id_k)
                                                vh_k = self.vehicle_helpers.get(id_k)
                                                print(
                                                    f"        {id_k}: halted={id_k in self.action_override} "
                                                    f"reason={self._halt_reason(id_k, k)} "
                                                    f"cas={self.bs.traf.cas[k]:.1f} "
                                                    f"intersection={getattr(vh_k, 'current_intersection', None)} "
                                                    f"route={getattr(vh_k, 'current_route_section', None)}"
                                                )
                                        else:
                                            over_cap = held_for >= self.max_slot_hold_steps
                                            print(
                                                f"    {route_section}: {level}ft, held {held_for} steps, "
                                                f"not draining (over_cap={over_cap}, cap={self.max_slot_hold_steps})"
                                            )
                                    queue = self.rr_queue.get(target_ni, [])
                                    print(f"    queue (waiting): {list(queue)}")
                            else:
                                # Different-route LOS: two DIFFERENT route
                                # sections converging on the same
                                # intersection should never share a level --
                                # that's the whole point of round-robin's
                                # slot allocation. Show what each route
                                # section is actually holding right now to
                                # check for a genuine allocation conflict
                                # rather than guessing at one.
                                shared_target = getattr(vh_i, "current_intersection", None) or getattr(vh_j, "current_intersection", None)
                                if shared_target in self.traffic_manager.intersections:
                                    print(f"[step {self.step_counter}] --- active slots at {shared_target} ---")
                                    active = self.rr_active.get(shared_target, {})
                                    granted = self.rr_granted_step.get(shared_target, {})
                                    draining = self.rr_draining.get(shared_target, {})
                                    for route_section, level in active.items():
                                        held_for = self.step_counter - granted.get(route_section, self.step_counter)
                                        drain_note = f"DRAINING(snapshot={sorted(draining[route_section])})" if route_section in draining else "not draining"
                                        flag = "  <-- " + ("rs_i" if route_section == rs_i else "rs_j" if route_section == rs_j else "") if route_section in (rs_i, rs_j) else ""
                                        print(f"    {route_section}: {level}ft, held {held_for} steps, {drain_note}{flag}")
                                    queue = self.rr_queue.get(shared_target, [])
                                    print(f"    queue (waiting): {list(queue)}")
                                    print(f"    rs_i={rs_i} rs_j={rs_j} -- both present in active above: {rs_i in active and rs_j in active}")

                                for label, id_x, vh_x in [("i", id_i, vh_i), ("j", id_j, vh_j)]:
                                    locked_rs = self.last_free_route_section.get(id_x)
                                    within = getattr(vh_x, "within_intersection", None)
                                    assigned = targets.get(id_x)
                                    live_grant_target = None
                                    live_grant_level = None
                                    for isec_id, active_dict in self.rr_active.items():
                                        if locked_rs in active_dict:
                                            live_grant_target = isec_id
                                            live_grant_level = active_dict[locked_rs]
                                            break
                                    stale = assigned is not None and (live_grant_level is None or live_grant_level != assigned)
                                    print(
                                        f"    [{label}] {id_x}: locked_route_section={locked_rs} "
                                        f"within_intersection={within} assigned_alt={assigned} "
                                        f"live_grant_at={live_grant_target}={live_grant_level} "
                                        f"STALE_FALLBACK={stale}"
                                    )
                            self.stop_after_los = True
        self.los_counter += len(current_los_pairs)
        self.prev_LOS_pairs = current_los_pairs

        for i in range(n_ac):
            id_ = self.bs.traf.id[i]
            # Remaining distance along this aircraft's active route.
            dist_to_go = geometries.geoms[i].length
            if dist_to_go < self.dGoal:
                self.bs.stack.stack("DEL {}".format(id_))
                if id_ in self.travel_start:
                    self.full_travel[id_] = self.bs.sim.simt - self.travel_start[id_]

                # Matches runner.py's store_data exactly: only reconciled
                # into nmacs/nmac_time when the aircraft actually completes
                # its route, not at episode end for still-in-flight ones --
                # same limitation as the reference, kept for parity.
                if id_ in self.acInfo and 1 in self.acInfo[id_]["NMAC"]:
                    # Materialize each sub-group immediately -- groupby's
                    # sub-iterators are invalidated once the outer iterator
                    # advances, so this can't be split into two passes.
                    nmac_groups = [(key, list(group)) for key, group in groupby(self.acInfo[id_]["NMAC"])]
                    group_keys = np.array([key for key, _ in nmac_groups])
                    self.nmac_time += sum(self.acInfo[id_]["NMAC"]) * self.simdt
                    self.nmacs += sum(group_keys)
                    # Raw per-event streak lengths (seconds) -- each
                    # individual NMAC event's own duration, separate from
                    # the scenario-level sum/count above.
                    for key, group_list in nmac_groups:
                        if key == 1:
                            self.nmac_event_lengths.append(len(group_list) * self.simdt)

    # ------------------------------------------------------------------
    # Episode driver
    # ------------------------------------------------------------------
    def run_one_iteration(self, scenario_file_override=None):
        """
        Runs a single episode and returns a ray object ref to the results
        dict, mirroring D2MAV_A/runner.py's Runner.run_one_iteration --
        that's what main_baseline.py's driver loop expects to call
        .remote() on and collect with ray.get(). scenario_file_override
        forces a specific scenario instead of the usual random draw.
        """
        self.reset(scenario_file_override=scenario_file_override)
        done = False
        while not done:
            done = self.step()

        if self.debug and not self.stop_after_los:
            n_ac = self.bs.traf.lat.shape[0]
            if n_ac > 0:
                print(f"[step {self.step_counter}] --- episode ended (max_steps) with {n_ac} aircraft still incomplete ---")
                for i in range(n_ac):
                    id_ = self.bs.traf.id[i]
                    vh = self.vehicle_helpers.get(id_)
                    print(
                        f"    {id_}: route={getattr(vh, 'current_route_section', None)} "
                        f"halted={id_ in self.action_override} "
                        f"reason={self._halt_reason(id_, i)} "
                        f"intersection={getattr(vh, 'current_intersection', None)} "
                        f"within_intersection={getattr(vh, 'within_intersection', None)} "
                        f"cas={self.bs.traf.cas[i]:.1f}"
                    )

        avg_noise_dict = {route_id: float(np.mean(vals)) for route_id, vals in self.average_noise_increase.items()}
        all_noise_samples = [v for vals in self.average_noise_increase.values() for v in vals]

        data = {
            "heuristic": self.heuristic,
            "scenario_file": self.scen_file_temp,
            "steps": self.step_counter,
            "los_counter": self.los_counter,
            "los_events": self.los_events,
            "los_same_route_events": self.los_same_route_count,
            "los_diff_route_events": self.los_diff_route_count,
            "nmacs": int(self.nmacs),          # matches D2MAV_A/runner.py's exact convention -- see _flag_nmacs
            "nmac_time": float(self.nmac_time),
            "nmac_event_lengths": self.nmac_event_lengths,  # RAW: one entry per individual NMAC streak (seconds)

            "halting_times": self.halting_times,             # RAW: one entry per individual halt event (seconds)
            "avg_halting_time": float(np.mean(self.halting_times)) if self.halting_times else 0.0,
            "midair_halting_times": self.midair_halting_times,  # RAW: same, midair-only halts
            "avg_midair_halting_time": float(np.mean(self.midair_halting_times)) if self.midair_halting_times else 0.0,

            "full_travel_times": self.full_travel,  # RAW: ac_id -> travel time (seconds), already per-aircraft
            "avg_travel_time": float(np.mean(list(self.full_travel.values()))) if self.full_travel else 0.0,
            "total_ac_spawned": len(self.travel_start),
            "total_ac_completed": len(self.full_travel),

            "airborne_count_history": self.airborne_count_history,  # RAW: one entry per simulated step
            "mean_airborne_count": float(np.mean(self.airborne_count_history)) if self.airborne_count_history else 0.0,
            "max_airborne_count": int(np.max(self.airborne_count_history)) if self.airborne_count_history else 0,

            # Levels traversed per individual climb event (descents excluded).
            # alt_adjustment_events is the flat, pooled version; full_alt_adjustments
            # keeps the per-aircraft breakdown (including aircraft with zero
            # climbs, as an empty list, so a per-aircraft mean isn't silently
            # biased by aircraft that never climbing being absent entirely).
            "alt_adjustment_events": self.alt_adjustment_events,  # RAW: flat list, one entry per climb event
            "full_alt_adjustments": self.full_alt_adjustments,    # RAW: ac_id -> [levels per climb event]
            "mean_alt_adjustment_levels": float(np.mean(self.alt_adjustment_events)) if self.alt_adjustment_events else 0.0,

            "max_noise_increase": float(self.max_noise_increase),
            "avg_noise_increase": self.average_noise_increase,  # RAW: route_id -> [individual per-step samples]
            "avg_noise_increase_means": avg_noise_dict,          # per-route MEAN, kept for quick reference only
            "mean_noise_increase": float(np.mean(all_noise_samples)) if all_noise_samples else 0.0,
        }
        return ray.put([data, self.id])