"""Open-world seeding: mining verified seed tasks from instructional videos (paper Sec. 2.2).

Harvests instructional videos with taxonomy-driven queries, scores CUA
suitability, and reconstructs executable seed trajectories via dual-path
(semantic + visual) boundary intersection followed by recheck, annotation,
and verification.
"""
