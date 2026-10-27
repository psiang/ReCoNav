# Third-party notices

ReCoNav code is licensed under Apache-2.0. The data package, downloaded scene files, upstream methods, model checkpoints and their dependencies retain their own licenses and terms.

The external baseline adapters interface with NavGPT, MapGPT, CA-Nav, GC-VLN, NavSpace/SNav, InternNav/DualVLN, Lang2LTL-2 and LIMP. Source URLs and exact revisions are recorded in `reconav/baselines/upstream.json`; available upstream license files are preserved under `reconav/baselines/licenses/`. The Lang2LTL-2 referring-expression prompt under `reconav/baselines/resources/` is an upstream asset, covered by the corresponding upstream terms.

The declared GroundingDINO patch preserves the CA-Nav experiment's detector phrase-to-class mapping. Apply it to the recorded dependency revision; its upstream license continues to apply.

HM3D/HM3DSem and scene-derived annotations/geometry are governed by the applicable Matterport terms, including academic-use and attribution requirements. Those terms are included by reference in the separately distributed data package. Code licensing does not grant additional rights to these data.

Core runtime dependencies include NumPy, Shapely/GEOS, SciPy, Trimesh, Pydantic, Z3, Pillow, OpenCV, plyfile and NetworkX. Installations retain the respective packages' original notices.
