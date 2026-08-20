# Apex Rig AI

Automatic rig generation for Blender, based on reference bones — Smart
detection, modular limbs, remap/retarget, and game-engine (FBX/GLTF) export.

**Apex Rig AI is an independent GPL fork of
[Auto-Rig Pro](https://blendermarket.com/products/auto-rig-pro) 3.77.33 by
Artell (Lucas Veber), maintained by Tony.** It is not affiliated with or
endorsed by Lucky3D / Artell. Original code copyright 2016–2024 Lucas Veber;
fork modifications copyright 2026 Tony. See [LICENSE.txt](LICENSE.txt).

## What differs from upstream

- Rebranded UI (panels, labels, exporter entries) as **Apex Rig AI**.
- Update checks and doc/tracker links point at this repository, not Lucky3D.
- Extension-safe preferences lookup (fixes a `KeyError` in the preset-path
  dialogs under Blender 4.2+ extension installs).
- Version detection derives the add-on's own package instead of a hardcoded
  install path, so any extension repo location works.
- Original CC0 icons (the upstream Royalty Free icons are not redistributable).

## What is deliberately missing

Auto-Rig Pro's Royalty Free assets — `limb_presets/modules.blend`,
`misc_presets/facial_setup.blend`, `misc_presets/humanoid.blend`, and the
original icon art — **cannot be freely redistributed** and are not in this
repository. Features that append from those files require an Auto-Rig Pro
license: buy it, then copy those files into the matching folders locally.
Everything else works without them.

## Install

1. Zip the repository contents (or download a release zip).
2. Blender: **Edit > Preferences > Add-ons > Install from Disk**.
3. Enable **Apex Rig AI**. The panel lives in **3D View > Sidebar**.

⚠️ Do not enable Apex Rig AI and Auto-Rig Pro at the same time — they
register the same operator/property names and will conflict. Enable one.

## License

Code: GPL-2.0-or-later (unchanged from upstream, copyright preserved).
Assets: CC0 / CC-BY-SA 4.0 as noted per folder. Full details in
[LICENSE.txt](LICENSE.txt).
