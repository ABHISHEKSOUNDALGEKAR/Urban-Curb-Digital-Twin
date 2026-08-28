# Model reference

Notation, units and the exact definition of every reported metric. The README
covers *why*; this covers *what precisely*.

## Units

| quantity | unit | note |
|---|---|---|
| simulation clock | minutes | SimPy `env.now` is always minutes |
| link travel time | seconds | link performance functions are conventionally in seconds; converted at the call site |
| distance | metres | except `vmt_miles`, which is named for its unit |
| money | US dollars | |
| occupancy | fraction of capacity in [0, 1] | |

## Entities

**Curb segment** — one face of one block. Carries a stall count, a posted price,
a time limit, a face length and a physical position. Its stalls are partitioned
between three regulation types; the partition can change during a run.

**Regulation type** — `passenger` (metered general parking), `delivery`
(commercial loading zone), `ridehail` (passenger loading / TNC zone). A vehicle
class may use one or more types; see the eligibility table in the README.

**Trip** — one vehicle's attempt to stop. Resolves to exactly one of:

| outcome | meaning |
|---|---|
| `parked` | found a legal stall inside its acceptable walking distance |
| `diverted` | gave up on the target area and took the nearest free stall anywhere in the district, accepting a long walk |
| `illegal` | double-parked or otherwise stopped illegally |
| `abandoned` | left the district without completing the activity |

## Metric definitions

Only trips that **arrived after warm-up** are counted.

| metric | definition |
|---|---|
| `passenger_search_time_min` | mean minutes between arriving at the destination block and securing a stall (or exhausting patience) |
| `passenger_search_time_p90_min` | 90th percentile of the same — the tail is what drivers actually complain about |
| `passenger_walk_distance_m` | mean round-trip walk between the stall and the destination |
| `passenger_abandonment_rate` | share of passenger trips resolving as `abandoned` |
| `delivery_delay_min` | mean of (search time + round-trip walk from the stall to the door); excludes the service itself, which is not a delay |
| `delivery_illegal_rate` | share of delivery trips resolving as `illegal` |
| `ridehail_wait_min` | mean minutes from a request being created to the passenger boarding — includes dispatch queueing, deadhead and curb search |
| `ridehail_circling_time_min` | mean minutes spent circling the block after a failed pickup-curb search |
| `illegal_parking_rate` | all illegal stops (including ridehail drop-offs) ÷ all trips |
| `curb_occupancy_<class>` | mean occupancy of that regulation type's stalls, sampled every 5 minutes between warm-up and the horizon |
| `curb_saturated_share` | share of samples in which the *metered* pool is ≥90% full — the regime in which cruising explodes; the district-wide figure hides this because slack in loading and TNC zones averages it away |
| `curb_turnover_per_stall_per_hour` | successful parking events ÷ total stalls ÷ observed hours |
| `vmt_miles` | total vehicle miles travelled by all classes |
| `cruising_vmt_share` | share of VMT accrued while searching for curb space |
| `congestion_index` | mean ratio of congested to free-flow travel time across links |
| `meter_revenue_usd` | meter payments; a transfer, excluded from social cost |
| `system_social_cost_usd` | `time_cost + vmt_cost + illegal_cost` (see below) |

### Social cost

```
time_cost    = Σ_class (search / delay / wait minutes) × value of time
vmt_cost     = vmt_miles × external cost per mile
illegal_cost = illegal events × social cost per event
```

Payments (meter revenue, fines) are transfers between road users and the city.
They move money; they do not consume resources. Including them would let a policy
that raises revenue by degrading service score as an improvement, so they are
reported separately and never enter the objective.

## Statistical reporting

Metrics are reported as a mean over independent seeds with a 95% confidence
interval (normal approximation; with n ≥ 10 the t correction is under 5%). A
scenario difference is flagged as distinguishable from noise only when the two
confidence intervals do not overlap — a conservative test, and deliberately so.
