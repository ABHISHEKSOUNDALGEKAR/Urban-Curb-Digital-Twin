"""Agent behaviour: curb choice, search, fallback ladders, class differences."""

from __future__ import annotations

import pytest

from src.agents.base_agent import SearchOutcome, curb_choice_score
from src.agents.delivery import DeliveryVehicle
from src.agents.passenger import PassengerCar
from src.agents.ridehail import RidehailFleet
from src.config import load_scenario
from src.simulation.environment import CurbWorld


@pytest.fixture
def world(fast_config):
    return CurbWorld(fast_config())


class TestCurbChoiceScore:
    def _params(self, **kw):
        base = {
            "alpha_walk_per_m": 0.01,
            "beta_price": 1.0,
            "gamma_search_per_min": 0.3,
            "delta_occupancy": 1.0,
        }
        base.update(kw)
        return base

    def test_more_walking_costs_more(self):
        p = self._params()
        near = curb_choice_score(p, walk_m=50, price=2, approach_min=1, occupancy=0.5)
        far = curb_choice_score(p, walk_m=250, price=2, approach_min=1, occupancy=0.5)
        assert far > near

    def test_higher_price_costs_more(self):
        p = self._params()
        cheap = curb_choice_score(p, walk_m=50, price=1, approach_min=1, occupancy=0.5)
        dear = curb_choice_score(p, walk_m=50, price=6, approach_min=1, occupancy=0.5)
        assert dear > cheap

    def test_fuller_block_costs_more(self):
        p = self._params()
        empty = curb_choice_score(p, walk_m=50, price=2, approach_min=1, occupancy=0.1)
        full = curb_choice_score(p, walk_m=50, price=2, approach_min=1, occupancy=0.95)
        assert full > empty

    def test_freight_is_more_distance_averse_than_passengers(self):
        """The behavioural distinction the whole model rests on."""
        cfg = load_scenario("baseline", seed=0)
        pas = cfg.agents["passenger"]
        dlv = cfg.agents["delivery"]
        assert dlv["alpha_walk_per_m"] > pas["alpha_walk_per_m"]
        assert dlv["max_walk_distance_m"] < pas["max_walk_distance_m"]

    def test_ridehail_values_search_time_most(self):
        cfg = load_scenario("baseline", seed=0)
        assert (
            cfg.agents["ridehail"]["gamma_search_per_min"]
            > cfg.agents["passenger"]["gamma_search_per_min"]
        )


class TestPassengerCar:
    def test_parks_when_space_is_available(self, world):
        agent = PassengerCar(world, world.nodes[0], 0.0)
        world.env.process(agent.run())
        world.env.run(until=400)
        assert agent.record.outcome in {SearchOutcome.PARKED, SearchOutcome.DIVERTED}
        assert agent.record.curb_id is not None

    def test_dwell_is_positive_and_finite(self, world):
        for _ in range(50):
            agent = PassengerCar(world, world.nodes[0], 0.0)
            assert 0 < agent.dwell_min < 1000

    def test_double_parks_when_district_is_full(self, fast_config):
        cfg = fast_config()
        world = CurbWorld(cfg)
        # Fill every stall so no legal option exists anywhere.
        for seg in world.inventory:
            for cls in ("passenger", "delivery", "ridehail"):
                while seg.occupy(cls, 0.0):
                    pass
        agent = PassengerCar(world, world.nodes[0], 0.0)
        agent.p = dict(agent.p, compliance_probability=0.0)  # force non-compliance
        world.env.process(agent.run())
        world.env.run(until=400)
        assert agent.record.outcome == SearchOutcome.ILLEGAL

    def test_compliant_driver_abandons_rather_than_parking_illegally(self, fast_config):
        world = CurbWorld(fast_config())
        for seg in world.inventory:
            for cls in ("passenger", "delivery", "ridehail"):
                while seg.occupy(cls, 0.0):
                    pass
        agent = PassengerCar(world, world.nodes[0], 0.0)
        agent.p = dict(agent.p, compliance_probability=1.0)
        world.env.process(agent.run())
        world.env.run(until=400)
        assert agent.record.outcome == SearchOutcome.ABANDONED

    def test_search_time_accumulates(self, world):
        agent = PassengerCar(world, world.nodes[4], 0.0)
        world.env.process(agent.run())
        world.env.run(until=400)
        assert agent.record.search_time_min >= 0.0


class TestDeliveryVehicle:
    def test_prefers_loading_zone_then_meter(self, world):
        agent = DeliveryVehicle(world, world.commercial_nodes[0], 0.0)
        assert agent.eligible_classes() == ("delivery", "passenger")

    def test_double_parks_when_no_zone_is_available(self, fast_config):
        world = CurbWorld(fast_config())
        for seg in world.inventory:
            for cls in ("passenger", "delivery", "ridehail"):
                while seg.occupy(cls, 0.0):
                    pass
        agent = DeliveryVehicle(world, world.commercial_nodes[0], 0.0)
        agent.p = dict(agent.p, illegal_parking_probability=1.0)
        world.env.process(agent.run())
        world.env.run(until=400)
        assert agent.record.outcome == SearchOutcome.ILLEGAL

    def test_double_parking_blocks_the_lane(self, fast_config):
        world = CurbWorld(fast_config())
        node = world.nodes[4]
        agent = DeliveryVehicle(world, node, 0.0)
        agent.double_park(node, 10.0)
        link = agent._double_parked_link
        blocked = world.router.link_travel_time_s(link, 0.0)
        agent.end_double_park()
        clear = world.router.link_travel_time_s(link, 0.0)
        assert blocked > clear

    def test_service_delay_excludes_service_itself(self, world):
        agent = DeliveryVehicle(world, world.commercial_nodes[0], 0.0)
        world.env.process(agent.run())
        world.env.run(until=400)
        assert agent.record.service_delay_min < agent.service_min + 60


class TestRidehail:
    def test_fleet_size_scales_with_demand(self, fast_config):
        small = RidehailFleet(CurbWorld(fast_config()), 100.0)
        large = RidehailFleet(CurbWorld(fast_config()), 600.0)
        assert large.size > small.size

    def test_fleet_respects_minimum(self, fast_config):
        fleet = RidehailFleet(CurbWorld(fast_config()), 1.0)
        assert fleet.size >= fast_config().agents["ridehail"]["fleet"]["min_vehicles"]

    def test_dispatch_assigns_nearest_idle_vehicle(self, fast_config):
        world = CurbWorld(fast_config())
        fleet = RidehailFleet(world, 60.0)
        for v in fleet.vehicles:
            v.node = world.nodes[-1]
        near = fleet.vehicles[0]
        near.node = world.nodes[0]
        req = fleet.submit(world.nodes[0], world.nodes[-1])
        assert fleet.assigned_vehicle(req) is near

    def test_requests_queue_when_fleet_is_busy(self, fast_config):
        world = CurbWorld(fast_config())
        fleet = RidehailFleet(world, 60.0)
        fleet.idle.clear()
        fleet.submit(world.nodes[0], world.nodes[1])
        assert fleet.queue_length == 1

    def test_short_dwell_relative_to_passenger(self):
        cfg = load_scenario("baseline", seed=0)
        rh_dwell_min = cfg.agents["ridehail"]["mean_dwell_s"] / 60.0
        assert rh_dwell_min < cfg.agents["passenger"]["mean_dwell_min"] / 10


class TestSearchMechanics:
    def test_search_stops_at_attempt_limit(self, fast_config):
        world = CurbWorld(fast_config())
        for seg in world.inventory:
            while seg.occupy("passenger", 0.0):
                pass
        agent = PassengerCar(world, world.nodes[0], 0.0)
        result: dict = {}

        def proc():
            seg, reg, walk = yield from agent.search_for_curb(world.nodes[0], ("passenger",))
            result["seg"] = seg

        world.env.process(proc())
        world.env.run(until=200)
        assert result["seg"] is None
        assert agent.record.failed_attempts <= int(world.common["max_search_attempts"])

    def test_candidates_respect_max_walk(self, world):
        agent = PassengerCar(world, world.nodes[0], 0.0)
        for _seg, walk, _reg in agent.rank_candidates(world.nodes[0], 1000.0, ("passenger",)):
            assert walk <= agent.max_walk_m() + 1e-6

    def test_zero_capacity_regulation_is_not_a_candidate(self, fast_config):
        cfg = fast_config(allocation={"passenger": 1.0, "delivery": 0.0, "ridehail": 0.0})
        world = CurbWorld(cfg)
        agent = DeliveryVehicle(world, world.commercial_nodes[0], 0.0)
        cands = agent.rank_candidates(agent.destination, 400.0, ("delivery",))
        assert cands == []
