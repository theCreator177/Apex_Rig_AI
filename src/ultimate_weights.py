# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# ***** END GPL LICENCE BLOCK *****

"""
Auto-Rig Pro -- ULTIMATE auto-weight engine.

This module adds the 'ULTIMATE' bind engine on top of Auto-Rig Pro's existing
binding pipeline. It does NOT compute weights from scratch: ARP's proven
Heat-Map bind path produces the initial skinning, and this module runs a
deterministic post-process + diagnostics pass over the resulting deform
vertex groups.

Post-process pipeline (per mesh, deform groups only):
    normalize -> cap influences -> prune small weights -> (optional) smooth
    -> re-normalize

The heavy lifting is done with a single BMesh deform-layer read/modify/write
per mesh, which is the correct and fast way to edit weights in bulk (the old
`rig_intelligence.py` prototype used an invalid `vg.foreach_set("add", ...)`
call -- that approach is intentionally not carried forward).

Public entry point used by auto_rig.py:
    post_process_weights(objects, armature, settings, report) -> dict

Operators:
    ARP_OT_ultimate_postprocess  -- re-run the post-process on selected meshes
    ARP_OT_ultimate_report       -- show the last diagnostics in a popup
"""

import bpy
import bmesh
import json
import math
import blf
import gpu
from gpu_extras.batch import batch_for_shader

from bpy.types import Operator
from bpy.props import IntProperty, FloatProperty, BoolProperty, StringProperty, EnumProperty
from mathutils import Vector
from mathutils.kdtree import KDTree
from bpy.app.handlers import persistent

# ARP utility layer (get_object, find_armature, get_skinned_objects, ...)
from .utils import *


# ---------------------------------------------------------------------------
# Settings container
# ---------------------------------------------------------------------------

class UltimateSettings:
    """Plain settings holder so the engine can be called with either the Blender
    Scene (reads arp_ult_* properties) or a hand-built object (tests)."""

    def __init__(self, normalize=True, max_influences=4, prune_threshold=0.01,
                 smooth=True, smooth_factor=0.5, smooth_iters=2):
        self.normalize = normalize
        self.max_influences = max_influences
        self.prune_threshold = prune_threshold
        self.smooth = smooth
        self.smooth_factor = smooth_factor
        self.smooth_iters = smooth_iters

    @classmethod
    def from_scene(cls, scn):
        return cls(
            normalize=getattr(scn, 'arp_ult_normalize', True),
            max_influences=getattr(scn, 'arp_ult_max_influences', 4),
            prune_threshold=getattr(scn, 'arp_ult_prune_threshold', 0.01),
            smooth=getattr(scn, 'arp_ult_smooth', True),
            smooth_factor=getattr(scn, 'arp_ult_smooth_factor', 0.5),
            smooth_iters=getattr(scn, 'arp_ult_smooth_iters', 2),
        )


def _coerce_settings(settings):
    """Accept a Scene, an UltimateSettings, or None and always return an
    UltimateSettings."""
    if settings is None:
        return UltimateSettings()
    if isinstance(settings, UltimateSettings):
        return settings
    # assume it's a Scene (or anything exposing arp_ult_* attributes)
    return UltimateSettings.from_scene(settings)


# ---------------------------------------------------------------------------
# Per-vertex weight maths (pure dict logic, easy to unit-test)
# ---------------------------------------------------------------------------

def normalize_vertex(weights):
    """Scale a {group_index: weight} dict so the values sum to 1.0.
    Returns (normalized_dict, was_unweighted). A vertex whose total is 0 is
    reported as unweighted and returned unchanged (never divided by zero)."""
    total = sum(weights.values())
    if total <= 0.0:
        return weights, True
    return {g: w / total for g, w in weights.items()}, False


def cap_influences(weights, max_n):
    """Keep only the `max_n` largest weights, drop the rest. max_n <= 0 means
    unlimited. Returns (capped_dict, removed_count)."""
    if max_n <= 0 or len(weights) <= max_n:
        return weights, 0
    # sort by weight descending, keep the strongest max_n
    kept = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:max_n]
    removed = len(weights) - len(kept)
    return dict(kept), removed


def prune_small(weights, threshold):
    """Drop weights strictly below `threshold`. Returns (pruned_dict, removed_count).
    If every weight is below the threshold the single largest is kept so the
    vertex never becomes accidentally unweighted."""
    if threshold <= 0.0:
        return weights, 0
    pruned = {g: w for g, w in weights.items() if w >= threshold}
    if not pruned and weights:
        g = max(weights, key=weights.get)
        pruned = {g: weights[g]}
    return pruned, len(weights) - len(pruned)


# ---------------------------------------------------------------------------
# Mesh-level helpers
# ---------------------------------------------------------------------------

def _deform_group_indices(obj, armature):
    """Return the set of vertex-group indices on `obj` that correspond to
    deforming bones of `armature`. Non-deform groups (UV pins, masks, custom
    user groups) are deliberately left untouched."""
    deform_bone_names = set()
    if armature and armature.type == 'ARMATURE':
        for b in armature.data.bones:
            if b.use_deform:
                deform_bone_names.add(b.name)

    indices = set()
    for vg in obj.vertex_groups:
        # If we could resolve the armature, restrict to its deform bones.
        # Otherwise (defensive) treat all groups as deform.
        if not deform_bone_names or vg.name in deform_bone_names:
            indices.add(vg.index)
    return indices


def smooth_pass(obj, settings):
    """Operator-based weight smoothing. Must run outside the BMesh block.
    Reuses the same vertex_group_smooth call ARP already relies on."""
    settings = _coerce_settings(settings)
    if not settings.smooth or settings.smooth_iters <= 0:
        return
    prev_active = bpy.context.view_layer.objects.active
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.object.vertex_group_smooth(
            group_select_mode='ALL',
            factor=settings.smooth_factor,
            repeat=settings.smooth_iters,
        )
    except Exception as err:
        print("[ULTIMATE] smooth skipped on", obj.name, ":", err)
    finally:
        if prev_active is not None:
            bpy.context.view_layer.objects.active = prev_active


def _process_mesh_weights(obj, deform_indices, settings):
    """Read deform weights through a BMesh deform layer, apply
    normalize/cap/prune, write them back. Returns a per-mesh diagnostics dict.
    Does NOT smooth (smoothing is an operator -> handled separately)."""
    settings = _coerce_settings(settings)
    me = obj.data

    bm = bmesh.new()
    bm.from_mesh(me)
    deform_layer = bm.verts.layers.deform.verify()

    diag = {
        'name': obj.name,
        'vert_count': len(bm.verts),
        'unweighted': [],
        'over_influence': 0,
        'max_influence': 0,
        'capped_removed': 0,
        'pruned_removed': 0,
    }

    for v in bm.verts:
        dvert = v[deform_layer]
        # only deform groups, ignore zero entries
        weights = {g: dvert[g] for g in list(dvert.keys())
                   if g in deform_indices and dvert[g] > 0.0}

        diag['max_influence'] = max(diag['max_influence'], len(weights))
        if settings.max_influences > 0 and len(weights) > settings.max_influences:
            diag['over_influence'] += 1

        # a vertex with no deform weight is unweighted, regardless of any
        # non-deform groups (UV pins, masks) it may carry
        if not weights:
            diag['unweighted'].append(v.index)

        # 1. normalize (zero-sum vertices are left unchanged by normalize_vertex)
        if settings.normalize:
            weights, _ = normalize_vertex(weights)

        # 2. cap influences
        weights, removed = cap_influences(weights, settings.max_influences)
        diag['capped_removed'] += removed

        # 3. prune small weights
        weights, pruned = prune_small(weights, settings.prune_threshold)
        diag['pruned_removed'] += pruned

        # 4. re-normalize after cap/prune so the survivors sum to 1.0
        if settings.normalize and weights:
            weights, _ = normalize_vertex(weights)

        # write back: clear deform groups we own, then set the new values
        for g in list(dvert.keys()):
            if g in deform_indices:
                del dvert[g]
        for g, w in weights.items():
            dvert[g] = w

    bm.to_mesh(me)
    bm.free()
    me.update()

    diag['unweighted_count'] = len(diag['unweighted'])
    # keep only a sample of indices in the report to avoid huge payloads
    diag['unweighted_sample'] = diag['unweighted'][:25]
    del diag['unweighted']
    return diag


def collect_diagnostics(per_mesh, armature, settings):
    """Roll up per-mesh diagnostics + armature-wide checks (zero-weight deform
    bones, left/right asymmetry)."""
    settings = _coerce_settings(settings)

    # which deform bones actually received weight anywhere?
    weighted_bone_names = set()

    objs = [get_object(d['name']) for d in per_mesh]
    objs = [o for o in objs if o is not None]

    # one pass per mesh to count influence per group (for zero-weight + asymmetry)
    group_influenced = {}  # group_name -> influenced vert count (across meshes)
    for o in objs:
        idx_to_name = {vg.index: vg.name for vg in o.vertex_groups}
        for v in o.data.vertices:
            for g in v.groups:
                if g.weight > 0.0:
                    name = idx_to_name.get(g.group)
                    if name is None:
                        continue
                    group_influenced[name] = group_influenced.get(name, 0) + 1

    deform_bone_names = []
    if armature and armature.type == 'ARMATURE':
        deform_bone_names = [b.name for b in armature.data.bones if b.use_deform]

    for name, cnt in group_influenced.items():
        if cnt > 0:
            weighted_bone_names.add(name)

    zero_weight_bones = [n for n in deform_bone_names
                         if n not in weighted_bone_names]

    # left/right asymmetry: compare influenced-vert counts of .l vs .r pairs
    asymmetry = []
    for name in group_influenced:
        if name.endswith('.l'):
            base = name[:-2]
            r_name = base + '.r'
            l_cnt = group_influenced.get(name, 0)
            r_cnt = group_influenced.get(r_name, 0)
            if r_cnt == 0 and l_cnt == 0:
                continue
            bigger = max(l_cnt, r_cnt)
            if bigger > 0:
                ratio = abs(l_cnt - r_cnt) / bigger
                if ratio > 0.25:  # >25% difference flagged
                    asymmetry.append({'pair': base, 'l': l_cnt, 'r': r_cnt,
                                      'ratio': round(ratio, 3)})

    report = {
        'per_mesh': per_mesh,
        'mesh_count': len(per_mesh),
        'total_verts': sum(d['vert_count'] for d in per_mesh),
        'unweighted_total': sum(d.get('unweighted_count', 0) for d in per_mesh),
        'over_influence_total': sum(d.get('over_influence', 0) for d in per_mesh),
        'capped_removed_total': sum(d.get('capped_removed', 0) for d in per_mesh),
        'pruned_removed_total': sum(d.get('pruned_removed', 0) for d in per_mesh),
        'max_influence_observed': max([d.get('max_influence', 0)
                                       for d in per_mesh], default=0),
        'zero_weight_bones': zero_weight_bones,
        'zero_weight_bone_count': len(zero_weight_bones),
        'asymmetry': asymmetry,
        'settings': {
            'normalize': settings.normalize,
            'max_influences': settings.max_influences,
            'prune_threshold': settings.prune_threshold,
            'smooth': settings.smooth,
        },
    }
    return report


# ---------------------------------------------------------------------------
# Orchestrator -- the public entry point called from auto_rig.py
# ---------------------------------------------------------------------------

def post_process_weights(objects, armature, settings, report=None):
    """Run the ULTIMATE post-process over `objects` (list of mesh objects or
    names). `armature` is the deforming armature (used to identify deform
    groups). `settings` may be a Scene or an UltimateSettings. `report` is an
    optional operator self.report callable for user-facing messages.

    Returns the diagnostics dict, and also stores it as JSON on
    scene.arp_ult_last_report when a Scene is the active context."""
    settings = _coerce_settings(settings)

    # resolve names -> objects, keep only meshes
    resolved = []
    for o in objects:
        obj = o if not isinstance(o, str) else get_object(o)
        if obj is not None and obj.type == 'MESH':
            resolved.append(obj)

    if not resolved:
        if report:
            report({'WARNING'}, "ULTIMATE: no mesh objects to post-process")
        return {'per_mesh': [], 'mesh_count': 0}

    # optional smoothing first (operator based), then deterministic edits
    if settings.smooth:
        for obj in resolved:
            smooth_pass(obj, settings)

    per_mesh = []
    for obj in resolved:
        deform_indices = _deform_group_indices(obj, armature)
        per_mesh.append(_process_mesh_weights(obj, deform_indices, settings))

    diag = collect_diagnostics(per_mesh, armature, settings)

    # persist + surface
    try:
        bpy.context.scene.arp_ult_last_report = json.dumps(diag)
    except Exception:
        pass

    if report:
        report({'INFO'},
               "ULTIMATE: {0} unweighted, {1} verts capped, {2} zero-weight bones"
               .format(diag.get('unweighted_total', 0),
                       diag.get('over_influence_total', 0),
                       diag.get('zero_weight_bone_count', 0)))
    print("[ULTIMATE] post-process report:", json.dumps(diag)[:2000])
    return diag


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Symmetrize Weights Logic
# ---------------------------------------------------------------------------

def _mirror_bone_name(name):
    """Swap L/R bone name suffixes."""
    for sfx_a, sfx_b in [('.l', '.r'), ('_l', '_r'), ('.L', '.R'), ('_L', '_R'),
                          ('_left', '_right'), ('_Left', '_Right')]:
        if name.endswith(sfx_a):
            return name[:-len(sfx_a)] + sfx_b
        if name.endswith(sfx_b):
            return name[:-len(sfx_b)] + sfx_a
    return name


def symmetrize_mesh_weights(obj, armature, direction='LEFT_TO_RIGHT', tolerance=1e-4):
    """Symmetrize vertex weights of obj based on armature deform groups."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    deform_layer = bm.verts.layers.deform.verify()
    
    deform_indices = _deform_group_indices(obj, armature)
    vg_sym_map = {}
    for idx in deform_indices:
        vg_name = obj.vertex_groups[idx].name
        mirror_name = _mirror_bone_name(vg_name)
        if mirror_name in obj.vertex_groups:
            vg_sym_map[idx] = obj.vertex_groups[mirror_name].index
        else:
            vg_sym_map[idx] = idx
            
    # Build KDTree
    num_verts = len(bm.verts)
    kd = KDTree(num_verts)
    for v in bm.verts:
        kd.insert(v.co, v.index)
    kd.balance()
    
    mirror_map = {}
    for v in bm.verts:
        if v.index in mirror_map:
            continue
        target = Vector((-v.co.x, v.co.y, v.co.z))
        co, idx, dist = kd.find(target)
        if dist < tolerance:
            mirror_map[v.index] = idx
            mirror_map[idx] = v.index
        else:
            mirror_map[v.index] = v.index
            
    def copy_mirrored(src_v, dst_v):
        src_d = src_v[deform_layer]
        dst_d = dst_v[deform_layer]
        for g in list(dst_d.keys()):
            if g in deform_indices:
                del dst_d[g]
        for g, w in src_d.items():
            if g in deform_indices and w > 0.0:
                g_sym = vg_sym_map[g]
                dst_d[g_sym] = w

    bm.verts.ensure_lookup_table()
    for v1 in bm.verts:
        v2_idx = mirror_map[v1.index]
        if v1.index == v2_idx:
            dvert = v1[deform_layer]
            weights = {g: w for g, w in dvert.items() if g in deform_indices}
            new_weights = {}
            visited = set()
            for g, w in weights.items():
                if g in visited:
                    continue
                g_sym = vg_sym_map[g]
                if g == g_sym:
                    new_weights[g] = w
                else:
                    w_sym = weights.get(g_sym, 0.0)
                    avg_w = 0.5 * (w + w_sym)
                    new_weights[g] = avg_w
                    new_weights[g_sym] = avg_w
                    visited.add(g)
                    visited.add(g_sym)
            for g in list(dvert.keys()):
                if g in deform_indices:
                    del dvert[g]
            for g, w in new_weights.items():
                if w > 0.0:
                    dvert[g] = w
            continue
            
        v2 = bm.verts[v2_idx]
        if v1.co.x < 0.0 and direction == 'LEFT_TO_RIGHT':
            copy_mirrored(v2, v1)
        elif v1.co.x > 0.0 and direction == 'RIGHT_TO_LEFT':
            copy_mirrored(v2, v1)
        elif direction == 'AVERAGE' and v1.co.x > 0.0:
            dvert1 = v1[deform_layer]
            dvert2 = v2[deform_layer]
            w1 = {g: dvert1.get(g, 0.0) for g in deform_indices}
            w2 = {g: dvert2.get(g, 0.0) for g in deform_indices}
            
            avg_weights_v1 = {}
            avg_weights_v2 = {}
            for g in deform_indices:
                g_sym = vg_sym_map[g]
                avg_w = 0.5 * (w1.get(g, 0.0) + w2.get(g_sym, 0.0))
                if avg_w > 0.0:
                    avg_weights_v1[g] = avg_w
                avg_w_sym = 0.5 * (w1.get(g_sym, 0.0) + w2.get(g, 0.0))
                if avg_w_sym > 0.0:
                    avg_weights_v2[g] = avg_w_sym
                    
            for g in list(dvert1.keys()):
                if g in deform_indices:
                    del dvert1[g]
            for g, w in avg_weights_v1.items():
                dvert1[g] = w
                
            for g in list(dvert2.keys()):
                if g in deform_indices:
                    del dvert2[g]
            for g, w in avg_weights_v2.items():
                dvert2[g] = w
                
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


# ---------------------------------------------------------------------------
# Auto Fix Unweighted Logic
# ---------------------------------------------------------------------------

def fix_mesh_unweighted_vertices(obj, armature):
    """Auto-fix unweighted vertices via topological BFS and KDTree spatial search."""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    deform_layer = bm.verts.layers.deform.verify()
    deform_indices = _deform_group_indices(obj, armature)
    
    weighted_verts = []
    unweighted_verts = []
    
    for v in bm.verts:
        dvert = v[deform_layer]
        w_sum = sum(dvert[g] for g in dvert.keys() if g in deform_indices)
        if w_sum > 0.0:
            weighted_verts.append(v)
        else:
            unweighted_verts.append(v)
            
    if not unweighted_verts:
        bm.free()
        return 0
        
    if not weighted_verts:
        bm.free()
        return 0
        
    kd = KDTree(len(weighted_verts))
    for i, wv in enumerate(weighted_verts):
        kd.insert(wv.co, wv.index)
    kd.balance()
    
    bm.verts.ensure_lookup_table()
    fixed_count = 0
    for uv in unweighted_verts:
        visited = {uv.index}
        found_weighted = []
        current_ring = [uv]
        
        while current_ring and not found_weighted:
            next_ring = []
            for v in current_ring:
                for edge in v.link_edges:
                    oth = edge.other_vert(v)
                    if oth.index not in visited:
                        visited.add(oth.index)
                        next_ring.append(oth)
                        oth_d = oth[deform_layer]
                        oth_w_sum = sum(oth_d[g] for g in oth_d.keys() if g in deform_indices)
                        if oth_w_sum > 0.0:
                            found_weighted.append(oth)
            current_ring = next_ring
            
        avg_weights = {}
        if found_weighted:
            for wv in found_weighted:
                wv_d = wv[deform_layer]
                wv_weights = {g: wv_d[g] for g in wv_d.keys() if g in deform_indices}
                total = sum(wv_weights.values())
                if total > 0.0:
                    for g, w in wv_weights.items():
                        avg_weights[g] = avg_weights.get(g, 0.0) + (w / total)
            total_avg = sum(avg_weights.values())
            if total_avg > 0.0:
                avg_weights = {g: w / total_avg for g, w in avg_weights.items()}
        else:
            co, idx, dist = kd.find(uv.co)
            wv = bm.verts[idx]
            wv_d = wv[deform_layer]
            wv_weights = {g: wv_d[g] for g in wv_d.keys() if g in deform_indices}
            total = sum(wv_weights.values())
            if total > 0.0:
                avg_weights = {g: w / total for g, w in wv_weights.items()}
                
        if avg_weights:
            uv_d = uv[deform_layer]
            for g in list(uv_d.keys()):
                if g in deform_indices:
                    del uv_d[g]
            for g, w in avg_weights.items():
                uv_d[g] = w
            fixed_count += 1
            
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return fixed_count


# ---------------------------------------------------------------------------
# GPU Drawing Helpers for Heatmap Legend
# ---------------------------------------------------------------------------

def _uniform_shader():
    name = 'UNIFORM_COLOR' if bpy.app.version >= (3, 4, 0) else '2D_UNIFORM_COLOR'
    return gpu.shader.from_builtin(name)


def _draw_rect(x, y, w, h, color):
    s = _uniform_shader()
    verts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    batch = batch_for_shader(s, 'TRIS', {"pos": verts}, indices=[(0, 1, 2), (0, 2, 3)])
    s.bind()
    s.uniform_float("color", color)
    batch.draw(s)


def _draw_rounded_rect(x, y, w, h, r, color, segments=8):
    s = _uniform_shader()
    r = max(0, min(r, w / 2, h / 2))
    if r < 1:
        _draw_rect(x, y, w, h, color)
        return
    cx, cy = x + w / 2, y + h / 2
    outline = []
    corners = [
        (x + r, y + r, math.pi, 1.5 * math.pi),
        (x + w - r, y + r, 1.5 * math.pi, 2.0 * math.pi),
        (x + w - r, y + h - r, 0, 0.5 * math.pi),
        (x + r, y + h - r, 0.5 * math.pi, math.pi),
    ]
    for (ccx, ccy, a_start, a_end) in corners:
        for i in range(segments + 1):
            t = i / segments
            a = a_start + (a_end - a_start) * t
            outline.append((ccx + r * math.cos(a), ccy + r * math.sin(a)))
    verts = [(cx, cy)] + outline
    n = len(outline)
    indices = [(0, i + 1, (i + 1) % n + 1) for i in range(n)]
    batch = batch_for_shader(s, 'TRIS', {"pos": verts}, indices=indices)
    s.bind()
    s.uniform_float("color", color)
    batch.draw(s)


def _draw_rounded_rect_outline(x, y, w, h, r, color, segments=8):
    s = _uniform_shader()
    r = max(0, min(r, w / 2, h / 2))
    outline = []
    if r < 1:
        outline = [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
    else:
        corners = [
            (x + r, y + r, math.pi, 1.5 * math.pi),
            (x + w - r, y + r, 1.5 * math.pi, 2.0 * math.pi),
            (x + w - r, y + h - r, 0, 0.5 * math.pi),
            (x + r, y + h - r, 0.5 * math.pi, math.pi),
        ]
        for (ccx, ccy, a_start, a_end) in corners:
            for i in range(segments + 1):
                t = i / segments
                a = a_start + (a_end - a_start) * t
                outline.append((ccx + r * math.cos(a), ccy + r * math.sin(a)))
        outline.append(outline[0])
    batch = batch_for_shader(s, 'LINE_STRIP', {"pos": outline})
    s.bind()
    s.uniform_float("color", color)
    batch.draw(s)


def _draw_circle(cx, cy, radius, color, segments=12):
    s = _uniform_shader()
    verts = [(cx, cy)]
    for i in range(segments + 1):
        a = 2 * math.pi * i / segments
        verts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    indices = [(0, i + 1, i + 2) for i in range(segments)]
    batch = batch_for_shader(s, 'TRIS', {"pos": verts}, indices=indices)
    s.bind()
    s.uniform_float("color", color)
    batch.draw(s)


def _draw_line(x1, y1, x2, y2, color, width=1.0):
    s = _uniform_shader()
    gpu.state.line_width_set(width)
    batch = batch_for_shader(s, 'LINES', {"pos": [(x1, y1), (x2, y2)]})
    s.bind()
    s.uniform_float("color", color)
    batch.draw(s)
    gpu.state.line_width_set(1.0)


def _draw_text(x, y, text, size=12, color=(1, 1, 1, 1)):
    fid = 0
    blf.position(fid, x, y, 0)
    if bpy.app.version >= (3, 4, 0):
        blf.size(fid, size)
    else:
        blf.size(fid, size, 72)
    blf.color(fid, *color)
    blf.draw(fid, text)


# ---------------------------------------------------------------------------
# Weight Quality Heatmap Viewport Overlay
# ---------------------------------------------------------------------------

class HeatmapOverlay:
    _handle_view = None
    _handle_pixel = None
    cache = {}

    @classmethod
    def is_active(cls):
        return cls._handle_view is not None

    @classmethod
    def draw_callback_view(cls, context):
        if not cls.is_active():
            return
        name = 'UNIFORM_COLOR' if bpy.app.version >= (3, 4, 0) else '3D_UNIFORM_COLOR'
        shader = gpu.shader.from_builtin(name)
        shader.bind()
        gpu.state.point_size_set(8.0)

        for obj_name, data in list(cls.cache.items()):
            obj = get_object(obj_name)
            if not obj or not obj.visible_get() or obj not in context.selected_objects:
                continue
            
            mw = obj.matrix_world
            for cat, color in [('unweighted', (1.0, 0.0, 1.0, 0.85)),
                               ('over', (1.0, 0.2, 0.0, 0.85)),
                               ('asym', (0.9, 0.8, 0.0, 0.70))]:
                cos = data.get(cat, [])
                if not cos:
                    continue
                wcos = [mw @ co for co in cos]
                batch = batch_for_shader(shader, 'POINTS', {"pos": wcos})
                shader.uniform_float("color", color)
                batch.draw(shader)

    @classmethod
    def draw_callback_pixel(cls, context):
        if not cls.is_active():
            return
        gpu.state.blend_set('ALPHA')
        x, y, w, h = 20, 20, 240, 100
        _draw_rounded_rect(x, y, w, h, 6, (0.10, 0.10, 0.12, 0.85))
        _draw_rounded_rect_outline(x, y, w, h, 6, (0.2, 0.2, 0.22, 0.9))
        
        _draw_text(x + 12, y + h - 22, "Ultimate Skin Health Legend", 11, (1.0, 1.0, 1.0, 1.0))
        _draw_line(x + 12, y + h - 28, x + w - 12, y + h - 28, (0.25, 0.25, 0.28, 0.6))
        
        _draw_circle(x + 20, y + 54, 5, (1.0, 0.0, 1.0, 1.0))
        _draw_text(x + 36, y + 49, "Unweighted Verts (0.0 sum)", 11, (0.9, 0.9, 0.9, 1.0))
        
        max_inf = context.scene.arp_ult_max_influences
        _draw_circle(x + 20, y + 36, 5, (1.0, 0.2, 0.0, 1.0))
        _draw_text(x + 36, y + 31, f"Over Influence Cap (>{max_inf})", 11, (0.9, 0.9, 0.9, 1.0))
        
        _draw_circle(x + 20, y + 18, 5, (0.9, 0.8, 0.0, 1.0))
        _draw_text(x + 36, y + 13, "Asymmetric Skinning (>0.1)", 11, (0.9, 0.9, 0.9, 1.0))
        gpu.state.blend_set('NONE')


def update_heatmap_cache(context):
    HeatmapOverlay.cache.clear()
    armature = None
    active = context.active_object
    if active and active.type == 'ARMATURE':
        armature = active
    else:
        for o in context.selected_objects:
            if o.type == 'MESH':
                arm = find_armature(o)
                if arm:
                    armature = arm
                    break
                    
    max_inf = context.scene.arp_ult_max_influences
    for obj in context.selected_objects:
        if obj.type != 'MESH':
            continue
            
        # build KDTree
        kd = KDTree(len(obj.data.vertices))
        for i, v in enumerate(obj.data.vertices):
            kd.insert(v.co, i)
        kd.balance()
        
        deform_indices = _deform_group_indices(obj, armature)
        vert_weights = {}
        for v in obj.data.vertices:
            dvert = {g.group: g.weight for g in v.groups if g.group in deform_indices and g.weight > 0.0}
            vert_weights[v.index] = dvert
            
        unweighted = []
        over = []
        asym = []
        
        mirror_map = {}
        tolerance = 1e-4
        for v in obj.data.vertices:
            if v.index in mirror_map:
                continue
            target = Vector((-v.co.x, v.co.y, v.co.z))
            co, idx, dist = kd.find(target)
            if dist < tolerance:
                mirror_map[v.index] = idx
                mirror_map[idx] = v.index
            else:
                mirror_map[v.index] = v.index
                
        vg_sym_map = {}
        for idx in deform_indices:
            vg_name = obj.vertex_groups[idx].name
            mirror_name = _mirror_bone_name(vg_name)
            if mirror_name in obj.vertex_groups:
                vg_sym_map[idx] = obj.vertex_groups[mirror_name].index
            else:
                vg_sym_map[idx] = idx
                
        for v in obj.data.vertices:
            w = vert_weights.get(v.index, {})
            w_sum = sum(w.values())
            if w_sum <= 0.0:
                unweighted.append(v.co.copy())
                continue
                
            if max_inf > 0 and len(w) > max_inf:
                over.append(v.co.copy())
                
            mv_idx = mirror_map[v.index]
            if mv_idx != v.index:
                mw = vert_weights.get(mv_idx, {})
                is_asymmetric = False
                for g, val in w.items():
                    g_sym = vg_sym_map.get(g)
                    m_val = mw.get(g_sym, 0.0)
                    if abs(val - m_val) > 0.1:
                        is_asymmetric = True
                        break
                if not is_asymmetric:
                    for g_sym, m_val in mw.items():
                        g = None
                        for orig_g, sym_g in vg_sym_map.items():
                            if sym_g == g_sym:
                                g = orig_g
                                break
                        val = w.get(g, 0.0) if g is not None else 0.0
                        if abs(val - m_val) > 0.1:
                            is_asymmetric = True
                            break
                if is_asymmetric:
                    asym.append(v.co.copy())
                    
        HeatmapOverlay.cache[obj.name] = {
            'unweighted': unweighted,
            'over': over,
            'asym': asym
        }


last_selected_names = set()
last_mode = None

@persistent
def depsgraph_update_handler(scene, depsgraph):
    global last_selected_names, last_mode
    if not HeatmapOverlay.is_active():
        return
    ctx = bpy.context
    selected_names = {o.name for o in ctx.selected_objects}
    mode = ctx.mode
    is_updated = False
    
    if selected_names != last_selected_names or mode != last_mode:
        is_updated = True
        last_selected_names = selected_names
        last_mode = mode
    else:
        # Avoid heavy KDTree rebuilds on Mesh evaluated updates during Pose/Object mode deforms
        if mode in {'EDIT', 'PAINT_WEIGHT'}:
            for update in depsgraph.updates:
                if isinstance(update.id, bpy.types.Mesh):
                    is_updated = True
                    break
    if is_updated:
        update_heatmap_cache(ctx)
        for area in ctx.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class ARP_OT_ultimate_postprocess(Operator):
    """Run the Ultimate weight post-process (normalize, cap, prune, smooth) on
    the selected skinned meshes, using the current Ultimate settings"""
    bl_idname = "arp.ultimate_postprocess"
    bl_label = "Ultimate Weight Post-Process"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        scn = context.scene
        targets = [o for o in context.selected_objects if o.type == 'MESH']
        armature = None

        if context.active_object and context.active_object.type == 'ARMATURE':
            armature = context.active_object
            if not targets:
                targets = [get_object(n) for n in get_skinned_objects(armature)]
        else:
            for o in targets:
                arm = find_armature(o)
                if arm:
                    armature = arm
                    break

        targets = [o for o in targets if o is not None]
        if not targets:
            self.report({'ERROR'}, "Select skinned mesh(es), or the armature")
            return {'CANCELLED'}

        post_process_weights(targets, armature,
                             UltimateSettings.from_scene(scn), self.report)
        # Update overlay cache if it is active
        if HeatmapOverlay.is_active():
            update_heatmap_cache(context)
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        return {'FINISHED'}


class ARP_OT_ultimate_symmetrize(Operator):
    """Symmetrize vertex weights of skinned meshes across X-axis"""
    bl_idname = "arp.ultimate_symmetrize"
    bl_label = "Symmetrize Weights"
    bl_options = {'REGISTER', 'UNDO'}

    direction: EnumProperty(
        name="Direction",
        items=(
            ('LEFT_TO_RIGHT', 'Left to Right (+X to -X)', ''),
            ('RIGHT_TO_LEFT', 'Right to Left (-X to +X)', ''),
            ('AVERAGE', 'Average Both Sides', ''),
        ),
        default='LEFT_TO_RIGHT'
    )
    tolerance: FloatProperty(name="Tolerance", default=1e-4, min=1e-6, max=1e-1, precision=6)

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        targets = [o for o in context.selected_objects if o.type == 'MESH']
        armature = None

        if context.active_object and context.active_object.type == 'ARMATURE':
            armature = context.active_object
            if not targets:
                targets = [get_object(n) for n in get_skinned_objects(armature)]
        else:
            for o in targets:
                arm = find_armature(o)
                if arm:
                    armature = arm
                    break

        targets = [o for o in targets if o is not None]
        if not targets:
            self.report({'ERROR'}, "Select skinned mesh(es), or the armature")
            return {'CANCELLED'}

        count = 0
        for obj in targets:
            symmetrize_mesh_weights(obj, armature, self.direction, self.tolerance)
            count += 1

        self.report({'INFO'}, f"Symmetrized weights on {count} meshes")
        # Update overlay cache if active
        if HeatmapOverlay.is_active():
            update_heatmap_cache(context)
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        return {'FINISHED'}


class ARP_OT_ultimate_fix_unweighted(Operator):
    """Automatically skin unweighted vertices from nearest weighted neighbors"""
    bl_idname = "arp.ultimate_fix_unweighted"
    bl_label = "Auto Fix Unweighted"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None

    def execute(self, context):
        targets = [o for o in context.selected_objects if o.type == 'MESH']
        armature = None

        if context.active_object and context.active_object.type == 'ARMATURE':
            armature = context.active_object
            if not targets:
                targets = [get_object(n) for n in get_skinned_objects(armature)]
        else:
            for o in targets:
                arm = find_armature(o)
                if arm:
                    armature = arm
                    break

        targets = [o for o in targets if o is not None]
        if not targets:
            self.report({'ERROR'}, "Select skinned mesh(es), or the armature")
            return {'CANCELLED'}

        total_fixed = 0
        for obj in targets:
            fixed = fix_mesh_unweighted_vertices(obj, armature)
            total_fixed += fixed

        self.report({'INFO'}, f"Fixed {total_fixed} unweighted vertices")
        # Update overlay cache if active
        if HeatmapOverlay.is_active():
            update_heatmap_cache(context)
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        return {'FINISHED'}


class ARP_OT_ultimate_heatmap_toggle(Operator):
    """Toggle weight quality heatmap overlay in 3D Viewport"""
    bl_idname = "arp.ultimate_heatmap_toggle"
    bl_label = "Toggle Heatmap Overlay"
    bl_options = {'REGISTER'}

    def execute(self, context):
        if HeatmapOverlay.is_active():
            try:
                bpy.types.SpaceView3D.draw_handler_remove(HeatmapOverlay._handle_view, 'WINDOW')
                bpy.types.SpaceView3D.draw_handler_remove(HeatmapOverlay._handle_pixel, 'WINDOW')
            except Exception:
                pass
            HeatmapOverlay._handle_view = None
            HeatmapOverlay._handle_pixel = None
            self.report({'INFO'}, "Heatmap Overlay disabled")
        else:
            update_heatmap_cache(context)
            HeatmapOverlay._handle_view = bpy.types.SpaceView3D.draw_handler_add(
                HeatmapOverlay.draw_callback_view, (context,), 'WINDOW', 'POST_VIEW')
            HeatmapOverlay._handle_pixel = bpy.types.SpaceView3D.draw_handler_add(
                HeatmapOverlay.draw_callback_pixel, (context,), 'WINDOW', 'POST_PIXEL')
            self.report({'INFO'}, "Heatmap Overlay enabled")

        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        return {'FINISHED'}


class ARP_OT_ultimate_report(Operator):
    """Show the diagnostics from the last Ultimate bind / post-process"""
    bl_idname = "arp.ultimate_report"
    bl_label = "Ultimate Weight Report"
    bl_options = {'REGISTER'}

    def execute(self, context):
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=420)

    def draw(self, context):
        layout = self.layout
        raw = getattr(context.scene, 'arp_ult_last_report', '')
        if not raw:
            layout.label(text="No Ultimate report yet. Bind with the Ultimate engine.")
            return
        try:
            diag = json.loads(raw)
        except Exception:
            layout.label(text="Report data could not be read.")
            return

        col = layout.column(align=True)
        col.label(text="Ultimate Weight Diagnostics", icon='MOD_VERTEX_WEIGHT')
        col.separator()
        col.label(text="Meshes processed: {0}".format(diag.get('mesh_count', 0)))
        col.label(text="Total vertices: {0}".format(diag.get('total_verts', 0)))
        col.label(text="Max influences observed: {0}".format(
            diag.get('max_influence_observed', 0)))

        col.separator()
        box = layout.box()
        bcol = box.column(align=True)
        unweighted = diag.get('unweighted_total', 0)
        bcol.label(text="Unweighted vertices: {0}".format(unweighted),
                   icon='ERROR' if unweighted else 'CHECKMARK')
        bcol.label(text="Verts over influence cap: {0}".format(
            diag.get('over_influence_total', 0)))
        bcol.label(text="Weights pruned: {0}".format(
            diag.get('pruned_removed_total', 0)))
        bcol.label(text="Influences capped: {0}".format(
            diag.get('capped_removed_total', 0)))

        zwb = diag.get('zero_weight_bones', [])
        col.separator()
        zbox = layout.box()
        zcol = zbox.column(align=True)
        zcol.label(text="Zero-weight deform bones: {0}".format(len(zwb)),
                   icon='BONE_DATA')
        for name in zwb[:12]:
            zcol.label(text="    " + name)
        if len(zwb) > 12:
            zcol.label(text="    ... and {0} more".format(len(zwb) - 12))

        asym = diag.get('asymmetry', [])
        if asym:
            col.separator()
            abox = layout.box()
            acol = abox.column(align=True)
            acol.label(text="L/R asymmetry warnings: {0}".format(len(asym)),
                       icon='MOD_MIRROR')
            for a in asym[:8]:
                acol.label(text="    {0}: L={1} R={2}".format(
                    a['pair'], a['l'], a['r']))


# ---------------------------------------------------------------------------
# Registration (operators + new Scene properties live here, NOT in auto_rig.py)
# ---------------------------------------------------------------------------

classes = (
    ARP_OT_ultimate_postprocess,
    ARP_OT_ultimate_symmetrize,
    ARP_OT_ultimate_fix_unweighted,
    ARP_OT_ultimate_heatmap_toggle,
    ARP_OT_ultimate_report
)


def register():
    from bpy.utils import register_class
    for cls in classes:
        try:
            register_class(cls)
        except Exception as err:
            print("[ULTIMATE] could not register", cls, err)

    Scn = bpy.types.Scene
    Scn.arp_ult_normalize = BoolProperty(
        name="Normalize Weights", default=True,
        description="Scale each vertex's deform weights so they sum to 1.0",
        options={'HIDDEN'})
    Scn.arp_ult_max_influences = IntProperty(
        name="Max Influences", default=4, min=0, max=12,
        description="Limit the number of bones influencing each vertex (0 = unlimited)."
                    "\n4 is the common limit for real-time game engines",
        options={'HIDDEN'})
    Scn.arp_ult_prune_threshold = FloatProperty(
        name="Prune Threshold", default=0.01, min=0.0, max=1.0, precision=4,
        description="Remove deform weights weaker than this value",
        options={'HIDDEN'})
    Scn.arp_ult_smooth = BoolProperty(
        name="Smooth Weights", default=True,
        description="Run a vertex weight smoothing pass before normalizing",
        options={'HIDDEN'})
    Scn.arp_ult_smooth_factor = FloatProperty(
        name="Smooth Factor", default=0.5, min=0.0, max=1.0,
        description="Strength of the smoothing pass", options={'HIDDEN'})
    Scn.arp_ult_smooth_iters = IntProperty(
        name="Smooth Iterations", default=2, min=0, max=20,
        description="Number of smoothing repetitions", options={'HIDDEN'})
    Scn.arp_ult_last_report = StringProperty(default='', options={'HIDDEN'})

    # Register depsgraph update post handler
    bpy.app.handlers.depsgraph_update_post.append(depsgraph_update_handler)


def unregister():
    # Remove depsgraph handler
    if depsgraph_update_handler in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(depsgraph_update_handler)

    # Disable overlay if active
    if HeatmapOverlay.is_active():
        try:
            bpy.types.SpaceView3D.draw_handler_remove(HeatmapOverlay._handle_view, 'WINDOW')
            bpy.types.SpaceView3D.draw_handler_remove(HeatmapOverlay._handle_pixel, 'WINDOW')
        except Exception:
            pass
        HeatmapOverlay._handle_view = None
        HeatmapOverlay._handle_pixel = None

    from bpy.utils import unregister_class
    for cls in reversed(classes):
        try:
            unregister_class(cls)
        except Exception:
            pass

    Scn = bpy.types.Scene
    for prop in ('arp_ult_normalize', 'arp_ult_max_influences',
                 'arp_ult_prune_threshold', 'arp_ult_smooth',
                 'arp_ult_smooth_factor', 'arp_ult_smooth_iters',
                 'arp_ult_last_report'):
        if hasattr(Scn, prop):
            try:
                delattr(Scn, prop)
            except Exception:
                pass
