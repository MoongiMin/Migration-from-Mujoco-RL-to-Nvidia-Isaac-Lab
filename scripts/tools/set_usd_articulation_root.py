#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", required=True, help="Path to USD file")
    parser.add_argument("--target", required=True, help="Prim path to set ArticulationRootAPI on")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app = AppLauncher(args).app
    from pxr import Usd, UsdPhysics

    usd_path = Path(args.usd).resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    print(f"[INFO] roots before: {[str(p.GetPath()) for p in roots]}")
    for prim in roots:
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)

    target = stage.GetPrimAtPath(args.target)
    if not target or not target.IsValid():
        raise RuntimeError(f"Target prim not found: {args.target}")
    UsdPhysics.ArticulationRootAPI.Apply(target)

    stage.GetRootLayer().Save()

    roots_after = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    print(f"[INFO] roots after: {[str(p.GetPath()) for p in roots_after]}")

    app.close()


if __name__ == "__main__":
    main()
