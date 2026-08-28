"""Urban Curb Digital Twin.

A multi-agent discrete-event simulation of competition for scarce urban curb
space between passenger vehicles, commercial freight and ridehail (TNC)
vehicles, with a calibration layer and a curb-allocation optimizer on top.

Package layout
--------------
``src.simulation``   world state: road graph, curb inventory, SimPy engine
``src.agents``       behavioural models for the three competing vehicle classes
``src.experiments``  scenario/seed orchestration, parallel execution, metrics
``src.calibration``  parameter estimation against observed occupancy
``src.optimization`` curb allocation objective, search algorithms, Pareto front
``src.sumo``         optional SUMO/TraCI traffic-physics backend
``src.viz``          static report generation and the Streamlit dashboard
"""

__version__ = "0.1.0"
