#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

from pxr import UsdPhysics


def main():
    usd_path = Path("source/isaaclab_assets/data/Robots/Nemo/nemo.usd").resolve()
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args.device)
    sim = SimulationContext(sim_cfg)

    cfg = sim_utils.UsdFileCfg(usd_path=str(usd_path))
    cfg.func("/World/Nemo", cfg)
    sim.reset()

    import omni.usd

    stage = omni.usd.get_context().get_stage()

    roots = []
    rigid_bodies = []
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if p.startswith("/World/Nemo"):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                roots.append(p)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_bodies.append(p)

    print(f"[INFO] Spawned USD: {usd_path}")
    print(f"[INFO] ArticulationRootAPI under /World/Nemo: {len(roots)}")
    for p in roots:
        print(f"  ROOT {p}")
    print(f"[INFO] RigidBodyAPI under /World/Nemo: {len(rigid_bodies)}")
    for p in rigid_bodies[:20]:
        print(f"  RB   {p}")
    if len(rigid_bodies) > 20:
        print(f"  ... {len(rigid_bodies) - 20} more")

    app.close()


if __name__ == "__main__":
    main()
