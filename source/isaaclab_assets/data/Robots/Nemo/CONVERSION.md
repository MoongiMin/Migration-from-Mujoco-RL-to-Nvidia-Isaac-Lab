# NEMO `nemo.usd` (MJCF → USD)

Isaac Lab expects:

`source/isaaclab_assets/data/Robots/Nemo/nemo.usd`

Source model: `legged_rl-main/legged_rl-main/models/nemo/nemo5.xml` with `assets/*.stl` next to it.

## Command-line (Isaac Lab)

From the Isaac Lab repo root:

```bat
isaaclab.bat -p scripts\tools\convert_mjcf.py --exit-after-convert --headless ^
  legged_rl-main\legged_rl-main\models\nemo\nemo5.xml ^
  source\isaaclab_assets\data\Robots\Nemo\nemo.usd
```

Or use the helper:

```bat
scripts\tools\convert_nemo_to_usd.bat
```

If the importer crashes in headless mode, try copying `nemo5.xml` and `assets\` to a **short path** (e.g. `C:\temp\nemo_mjcf\`) and run `convert_mjcf.py` with those paths. You can also run **without** `--headless` (loads the full GUI kit) and keep `--exit-after-convert` so the app exits after writing the USD.

## Isaac Sim GUI

Use **File → Import** and the MJCF importer, then **Export** the result as USD to `nemo.usd` at the path above.

## Notes

- On some Windows setups, MJCF conversion can hit driver/RTX or file-rename issues; using a short output path and ensuring no other process locks mesh folders can help.
- Joint names in the exported USD should match `NEMO_CFG` in `isaaclab_assets/robots/nemo.py`; rename in Sim if the importer changes them.

## If import / `convert_mjcf` crashes with “Corrupt asset … *.tmp.usd”

The MJCF importer writes mesh intermediates under `assets/<name>_tmp/` (e.g. `l_knee_tmp/l_knee.tmp.usd`). A failed conversion can leave **half-written** files; the next run then errors with **Corrupt asset** and may crash.

**Fix:** delete every `*_tmp` folder under `models/nemo/assets/` (and under `C:\temp\nemo_mjcf\assets` if you use a short-path copy), then rerun import or `convert_mjcf.py`. Only STL files should remain in `assets/` alongside `nemo5.xml`.
