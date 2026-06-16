"""Experiments — implemented Week 11-14.

Solver sweep (W18–W23): 3 solver × K∈{1,3} → Table I/II (stats_analysis).
exp3_phase_transition removed 2026-06-14 — severity is fixed per episode, no
phase transitions to test. exp8_aoi removed 2026-06-14 — its LCFS-vs-FCFS
ablation compared the 4 retired HR/SpO2/BP/Temperature streams, obsoleted by
the F=4→F=1 consolidated "ambulance_status" AoI stream.
"""
