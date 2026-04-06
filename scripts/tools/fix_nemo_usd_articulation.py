#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ensure a USD has exactly one ArticulationRootAPI prim.

This script launches Isaac Sim minimally (via AppLauncher) so pxr APIs are available.

Usage:
    isaaclab.bat -p scripts/tools/fix_nemo_usd_articulation.py --headless --usd <path> --keep <prim-path> --apply
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", required=True, help="Path to USD file")
    parser.add_argument(
        "--keep",
        default="",
        help="Prim path to keep as articulation root (e.g. /worldBody). If empty, keep first found.",
    )
    parser.add_argument("--apply", action="store_true", help="Apply and save changes")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    _simulation_app = app_launcher.app

    from pxr import Usd, UsdPhysics

    usd_path = Path(args.usd).expanduser().resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        raise RuntimeError(f"Failed to open USD: {usd_path}")

    roots = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    print(f"[INFO] USD: {usd_path}")
    print(f"[INFO] ArticulationRootAPI count: {len(roots)}")
    for prim in roots:
        print(f"  - {prim.GetPath()}")

    if len(roots) <= 1:
        print("[INFO] No fix needed.")
        _simulation_app.close()
        return

    keep_prim = None
    if args.keep:
        keep_token = args.keep.strip()
        # allow suffix matching if user passes /worldBody while stage path is /World/Nemo/worldBody
        matches = [p for p in roots if str(p.GetPath()) == keep_token or str(p.GetPath()).endswith(keep_token)]
        if matches:
            keep_prim = matches[0]
        else:
            print(f"[WARN] Requested keep path not found among roots: {keep_token}. Falling back to first root.")
    if keep_prim is None:
        keep_prim = roots[0]

    print(f"[INFO] Keeping root: {keep_prim.GetPath()}")
    for prim in roots:
        if prim == keep_prim:
            continue
        if args.apply:
            ok = prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            print(f"[INFO] Remove ArticulationRootAPI from {prim.GetPath()} -> {ok}")
        else:
            print(f"[DRYRUN] Would remove ArticulationRootAPI from {prim.GetPath()}")

    if args.apply:
        stage.GetRootLayer().Save()
        print("[INFO] USD saved.")
    else:
        print("[INFO] Dry-run only. Use --apply to save.")

    _simulation_app.close()


if __name__ == "__main__":
    main()
