#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", required=True, help="USD file path")
    parser.add_argument("--out", required=True, help="Output txt path")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(args).app
    from pxr import Usd, UsdPhysics

    usd_path = Path(args.usd).resolve()
    out_path = Path(args.out).resolve()
    stage = Usd.Stage.Open(str(usd_path))
    lines = [f"usd: {usd_path}"]
    if not stage:
        lines.append("stage_open: false")
    else:
        lines.append("stage_open: true")
        roots = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
        rigid_bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        lines.append(f"articulation_roots: {len(roots)}")
        for prim in roots:
            lines.append(str(prim.GetPath()))
        lines.append(f"rigid_bodies: {len(rigid_bodies)}")
        for prim in rigid_bodies[:20]:
            lines.append(f"rb:{prim.GetPath()}")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    app.close()


if __name__ == "__main__":
    main()
