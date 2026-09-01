# Dashboard source

The Streamlit layout, choropleth, scenario tabs, and control wiring come from:

https://github.com/ayushdabra/GeoMind

Live prototype the collaborator shared: https://oasis-geomind.streamlit.app/

This folder keeps the UI. Placeholder `tools.py` / `agent.py` solvers from that repo were **not** copied as the live path. Allocation, population, SIMD, forecast, and GeoShapley are served from `src/agent/dashboard_bridge.py` and the existing OASIS pipeline.
