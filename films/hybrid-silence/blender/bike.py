"""Procedural naked-street hybrid motorcycle + cinematic shot rig (Blender 4.x, EEVEE).

  blender -b --python-exit-code 1 -P bike.py -- --shot reveal --out renders/reveal
      [--size 720x1280] [--samples 24] [--still 1]

Shots: reveal, tank, smg, launch, profile, hero. Each sets its own camera/light
animation and frame range (30 fps). Units are metres; X is forward, Z is up.
"""

import argparse
import math
import os
import sys

import bmesh
import bpy
from mathutils import Vector

FPS = 30


# ── helpers ──────────────────────────────────────────────────────────────────
def mat(name, color, metallic=0.0, rough=0.5, coat=0.0, emit=None, strength=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = rough
    if coat:
        b.inputs["Coat Weight"].default_value = coat
        b.inputs["Coat Roughness"].default_value = 0.05
    if emit:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = strength
    return m


def assign(obj, m):
    obj.data.materials.clear()
    obj.data.materials.append(m)
    return obj


def smooth(obj, bevel=0.0, segs=3):
    for p in obj.data.polygons:
        p.use_smooth = True
    if bevel:
        mod = obj.modifiers.new("bev", "BEVEL")
        mod.width, mod.segments, mod.limit_method = bevel, segs, "ANGLE"
    return obj


def box(name, loc, size, m, rot=(0, 0, 0), bevel=0.012):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=[math.radians(r) for r in rot])
    o = bpy.context.active_object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(scale=True)
    return assign(smooth(o, bevel), m)


def ellipsoid(name, loc, size, m, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, location=loc, rotation=[math.radians(r) for r in rot])
    o = bpy.context.active_object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(scale=True)
    return assign(smooth(o), m)


def cyl(name, a, b, r, m, verts=32):
    a, b = Vector(a), Vector(b)
    d = b - a
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=d.length, location=(a + b) / 2)
    o = bpy.context.active_object
    o.name = name
    o.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
    return assign(smooth(o, 0.003, 2), m)


def disc(name, loc, r, depth, m, verts=48):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=depth, location=loc, rotation=(math.radians(90), 0, 0))
    o = bpy.context.active_object
    o.name = name
    return assign(smooth(o, 0.002, 2), m)


def tube(name, pts, r, m):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions, cu.bevel_depth, cu.bevel_resolution = "3D", r, 4
    sp = cu.splines.new("BEZIER")
    sp.bezier_points.add(len(pts) - 1)
    for bp, p in zip(sp.bezier_points, pts):
        bp.co = p
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    o = bpy.data.objects.new(name, cu)
    bpy.context.collection.objects.link(o)
    o.data.materials.append(m)
    return o


def arc(name, center, radius, a0, a1, width, thick, m):
    """Curved fender: rectangular section swept along an XZ arc (degrees, 0 = +X)."""
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    rings = []
    n = 32
    for i in range(n + 1):
        t = math.radians(a0 + (a1 - a0) * i / n)
        c, s = math.cos(t), math.sin(t)
        ring = []
        for dr, dy in ((0, -width), (thick, -width), (thick, width), (0, width)):
            rr = radius + dr
            ring.append(bm.verts.new((center[0] + rr * c, center[1] + dy, center[2] + rr * s)))
        rings.append(ring)
    for r0, r1 in zip(rings, rings[1:]):
        for k in range(4):
            bm.faces.new((r0[k], r0[(k + 1) % 4], r1[(k + 1) % 4], r1[k]))
    bm.faces.new(rings[0][::-1])
    bm.faces.new(rings[-1])
    bm.to_mesh(me)
    bm.free()
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    return assign(smooth(o, 0.004, 2), m)


def tft_mat():
    m = mat("tft", (0.004, 0.006, 0.01), rough=0.08, emit=(1, 1, 1), strength=0.0)
    tex = m.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tft.png"))
    m.node_tree.links.new(tex.outputs["Color"], m.node_tree.nodes["Principled BSDF"].inputs["Emission Color"])
    return m


def screen(name, center, m, half_w=0.11, half_h=0.068, tilt=60):
    """UV-mapped display quad facing the rider (normal tilted back by `tilt` degrees)."""
    t = math.radians(tilt)
    n = Vector((-math.sin(t), 0, math.cos(t)))
    up = Vector((math.cos(t), 0, math.sin(t)))
    left = Vector((0, 1, 0))  # rider faces +X, so +Y is their left
    c = Vector(center) + n * 0.014
    corners = [c + left * half_w - up * half_h, c - left * half_w - up * half_h, c - left * half_w + up * half_h, c + left * half_w + up * half_h]
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in corners], [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new()
    for loop, co in zip(uv.data, ((0, 0), (1, 0), (1, 1), (0, 1))):
        loop.uv = co
    o = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(o)
    return assign(o, m)


def parent_all(objs, parent):
    bpy.context.view_layer.update()
    for o in objs:
        o.parent = parent
        o.matrix_parent_inverse = parent.matrix_world.inverted()


# ── the motorcycle ───────────────────────────────────────────────────────────
def build_bike():
    M = {
        "paint": mat("paint", (0.010, 0.045, 0.22), metallic=0.7, rough=0.22, coat=1.0),
        "accent": mat("accent", (0.0, 0.42, 0.62), metallic=0.8, rough=0.25, coat=1.0),
        "black": mat("black", (0.012, 0.012, 0.014), rough=0.55),
        "gloss": mat("gloss", (0.008, 0.008, 0.01), rough=0.15, coat=1.0),
        "metal": mat("metal", (0.035, 0.035, 0.04), metallic=1.0, rough=0.35),
        "silver": mat("silver", (0.55, 0.56, 0.6), metallic=1.0, rough=0.22),
        "chrome": mat("chrome", (0.92, 0.92, 0.95), metallic=1.0, rough=0.06),
        "rubber": mat("rubber", (0.004, 0.004, 0.004), rough=0.7),
        "gold": mat("gold", (0.85, 0.55, 0.12), metallic=1.0, rough=0.25),
        "drl": mat("drl", (0.8, 0.95, 1.0), emit=(0.75, 0.93, 1.0), strength=0.0),
        "lens": mat("lens", (0.9, 0.95, 1.0), rough=0.05, emit=(1.0, 0.97, 0.9), strength=0.0),
        "tail": mat("tail", (0.3, 0.0, 0.0), emit=(1.0, 0.05, 0.03), strength=0.0),
        "smg": mat("smg", (0.0, 0.3, 0.4), emit=(0.0, 0.85, 1.0), strength=0.0),
        "tft": tft_mat(),
    }
    parts = []
    W = []  # wheel spinners

    for wx, tag in ((0.67, "f"), (-0.66, "r")):
        spin = bpy.data.objects.new(f"spin_{tag}", None)
        spin.location = (wx, 0, 0.30)
        bpy.context.collection.objects.link(spin)
        bpy.ops.mesh.primitive_torus_add(major_radius=0.248, minor_radius=0.058, major_segments=96, minor_segments=24, location=(wx, 0, 0.30), rotation=(math.radians(90), 0, 0))
        tire = assign(smooth(bpy.context.active_object), M["rubber"])
        tire.scale = (1, 1, 0.82 if tag == "f" else 1.05)
        bpy.ops.mesh.primitive_torus_add(major_radius=0.205, minor_radius=0.011, major_segments=96, minor_segments=12, location=(wx, 0, 0.30), rotation=(math.radians(90), 0, 0))
        rim = assign(smooth(bpy.context.active_object), M["gloss"])
        hub = disc(f"hub_{tag}", (wx, 0, 0.30), 0.045, 0.14, M["silver"])
        spokes = []
        for k in range(5):
            t = math.radians(k * 72 + 18)
            for off in (-8, 8):
                t2 = t + math.radians(off)
                a = (wx + 0.045 * math.cos(t), 0, 0.30 + 0.045 * math.sin(t))
                b = (wx + 0.2 * math.cos(t2), 0, 0.30 + 0.2 * math.sin(t2))
                spokes.append(cyl(f"spoke_{tag}{k}{off}", a, b, 0.009, M["gloss"], 12))
        rotor = disc(f"rotor_{tag}", (wx, 0.075, 0.30), 0.14 if tag == "f" else 0.11, 0.005, M["chrome"])
        parent_all([tire, rim, hub, rotor, *spokes], spin)
        parts += [tire, rim, hub, rotor, *spokes]
        W.append(spin)
        # caliper (not spinning)
        parts.append(box(f"caliper_{tag}", (wx + (0.08 if tag == "f" else -0.07), 0.085, 0.40 if tag == "f" else 0.38), (0.08, 0.035, 0.05), M["gold"], rot=(0, -40 if tag == "f" else 40, 0)))

    # front end: forks, clamps, headlight, bars
    for y in (-0.085, 0.085):
        parts.append(cyl(f"fork_low{y}", (0.67, y, 0.30), (0.585, y, 0.62), 0.028, M["gloss"]))
        parts.append(cyl(f"fork_up{y}", (0.585, y, 0.62), (0.50, y, 0.98), 0.02, M["chrome"]))
    parts.append(box("clamp_top", (0.505, 0, 0.98), (0.07, 0.22, 0.03), M["metal"], rot=(0, -22, 0)))
    parts.append(arc("fender_f", (0.67, 0, 0.30), 0.315, 20, 125, 0.06, 0.014, M["paint"]))
    parts.append(box("hl_housing", (0.63, 0, 0.84), (0.12, 0.2, 0.15), M["black"], rot=(0, -18, 0), bevel=0.03))
    parts.append(box("hl_visor", (0.60, 0, 0.93), (0.13, 0.17, 0.05), M["paint"], rot=(0, -35, 0), bevel=0.02))
    for y in (-0.055, 0.055):
        parts.append(box(f"drl{y}", (0.695, y, 0.86), (0.012, 0.065, 0.012), M["drl"], rot=(0, -18, 30 if y > 0 else -30), bevel=0.004))
    parts.append(ellipsoid("hl_lens", (0.69, 0, 0.81), (0.02, 0.05, 0.035), M["lens"]))
    parts.append(box("tft_housing", (0.535, 0, 1.085), (0.025, 0.26, 0.17), M["gloss"], rot=(0, -150, 0), bevel=0.01))
    parts.append(screen("tft", (0.535, 0, 1.085), M["tft"]))
    parts.append(tube("bars", [(0.44, -0.37, 1.07), (0.48, -0.18, 1.03), (0.49, 0, 1.02), (0.48, 0.18, 1.03), (0.44, 0.37, 1.07)], 0.012, M["black"]))
    for y in (-0.36, 0.36):
        parts.append(cyl(f"grip{y}", (0.445, y * 0.97, 1.068), (0.435, y * 1.12, 1.075), 0.019, M["rubber"]))
        parts.append(cyl(f"mirror_stem{y}", (0.46, y * 0.72, 1.05), (0.43, y * 0.86, 1.17), 0.005, M["black"]))
        parts.append(ellipsoid(f"mirror{y}", (0.43, y * 0.9, 1.19), (0.012, 0.05, 0.03), M["black"]))

    # tank, shrouds, seat, tail
    parts.append(ellipsoid("tank", (0.20, 0, 0.88), (0.30, 0.165, 0.125), M["paint"], rot=(0, -6, 0)))
    parts.append(box("tank_spine", (0.20, 0, 1.0), (0.32, 0.022, 0.012), M["accent"], rot=(0, -4, 0), bevel=0.004))
    for s in (-1, 1):
        parts.append(box(f"shroud{s}", (0.36, s * 0.14, 0.80), (0.26, 0.028, 0.17), M["paint"], rot=(s * 6, -24, s * 4), bevel=0.02))
        parts.append(box(f"shroud_acc{s}", (0.39, s * 0.157, 0.76), (0.18, 0.012, 0.035), M["accent"], rot=(s * 6, -24, s * 4), bevel=0.005))
        parts.append(box(f"sidecover{s}", (-0.30, s * 0.11, 0.76), (0.28, 0.02, 0.08), M["paint"], rot=(0, 10, 0), bevel=0.012))
    parts.append(box("seat", (-0.20, 0, 0.84), (0.40, 0.25, 0.10), M["black"], rot=(0, 5, 0), bevel=0.035))
    parts.append(box("pillion", (-0.50, 0, 0.90), (0.24, 0.2, 0.09), M["black"], rot=(0, 12, 0), bevel=0.03))
    parts.append(box("tail", (-0.64, 0, 0.90), (0.30, 0.15, 0.11), M["paint"], rot=(0, 16, 0), bevel=0.025))
    parts.append(box("taillight", (-0.79, 0, 0.945), (0.014, 0.11, 0.025), M["tail"], rot=(0, 16, 0), bevel=0.004))
    parts.append(arc("hugger", (-0.66, 0, 0.30), 0.315, 60, 150, 0.055, 0.014, M["black"]))

    # frame, engine, SMG, exhaust, swingarm, shock
    for y in (-0.115, 0.115):
        parts.append(tube(f"frame{y}", [(0.47, y * 0.7, 0.92), (0.33, y, 0.70), (0.22, y, 0.52), (-0.12, y, 0.52), (-0.62, y * 0.7, 0.86)], 0.017, M["gloss"]))
        parts.append(box(f"swing{y}", (-0.39, y * 1.05, 0.39), (0.56, 0.025, 0.055), M["gloss"], rot=(0, -17, 0), bevel=0.01))
        parts.append(cyl(f"peg{y}", (-0.12, y, 0.40), (-0.12, y * 2.1, 0.40), 0.012, M["chrome"]))
    parts.append(box("crankcase", (0.05, 0, 0.44), (0.38, 0.22, 0.26), M["metal"], bevel=0.035))
    parts.append(box("cylinder", (0.22, 0, 0.56), (0.15, 0.16, 0.2), M["black"], rot=(0, -20, 0), bevel=0.015))
    for i in range(6):
        parts.append(box(f"fin{i}", (0.20 + i * 0.012, 0, 0.50 + i * 0.03), (0.17, 0.2, 0.008), M["silver"], rot=(0, -20, 0), bevel=0.002))
    parts.append(disc("smg_cover", (0.0, 0.125, 0.42), 0.095, 0.03, M["metal"]))
    bpy.ops.mesh.primitive_torus_add(major_radius=0.09, minor_radius=0.006, major_segments=64, minor_segments=8, location=(0.0, 0.142, 0.42), rotation=(math.radians(90), 0, 0))
    smg_ring = assign(bpy.context.active_object, M["smg"])
    smg_ring.name = "smg_ring"
    parts.append(smg_ring)
    parts.append(box("muffler", (-0.06, 0.05, 0.27), (0.34, 0.14, 0.10), M["black"], rot=(0, 6, 0), bevel=0.025))
    parts.append(cyl("exhaust_tip", (-0.23, 0.08, 0.29), (-0.28, 0.08, 0.295), 0.024, M["chrome"]))
    parts.append(tube("header", [(0.28, 0.06, 0.50), (0.33, 0.07, 0.36), (0.22, 0.06, 0.28), (0.10, 0.05, 0.27)], 0.02, M["metal"]))
    parts.append(cyl("shock", (-0.28, 0, 0.44), (-0.16, 0, 0.74), 0.022, M["gold"]))
    parts.append(tube("chain", [(0.02, 0.1, 0.36), (-0.3, 0.1, 0.34), (-0.66, 0.1, 0.33)], 0.008, M["black"]))

    root = bpy.data.objects.new("bike", None)
    bpy.context.collection.objects.link(root)
    parent_all([o for o in parts + W if o.parent is None], root)
    return root, W, M


# ── stage ────────────────────────────────────────────────────────────────────
def build_stage():
    world = bpy.data.worlds.new("w")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.002, 0.003, 0.006, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    haze = world.node_tree.nodes.new("ShaderNodeVolumeScatter")
    haze.inputs["Density"].default_value = 0.0035
    haze.inputs["Anisotropy"].default_value = 0.35
    world.node_tree.links.new(haze.outputs[0], world.node_tree.nodes["World Output"].inputs["Volume"])

    bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0.0))
    floor = bpy.context.active_object
    fm = mat("floor", (0.006, 0.007, 0.009), metallic=0.2, rough=0.28)
    assign(floor, fm)

    def area(name, loc, energy, size, color, target=(0, 0, 0.6)):
        L = bpy.data.lights.new(name, "AREA")
        L.energy, L.size, L.color = energy, size, color
        o = bpy.data.objects.new(name, L)
        o.location = loc
        bpy.context.collection.objects.link(o)
        o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        return o

    lights = {
        "rim_l": area("rim_l", (-2.2, 1.8, 1.6), 600, 1.2, (0.55, 0.8, 1.0)),
        "rim_r": area("rim_r", (-2.2, -1.8, 1.6), 600, 1.2, (0.55, 0.8, 1.0)),
        "top": area("top", (0.1, 0, 3.4), 60, 1.6, (0.9, 0.95, 1.0)),
        "key": area("key", (2.2, -1.6, 1.4), 0, 1.0, (1.0, 0.9, 0.8)),
        "sweep": area("sweep", (2.0, -2.2, 1.2), 0, 0.35, (1.0, 1.0, 1.0)),
    }
    for k in ("rim_l", "rim_r", "sweep"):  # edge lights only; their soft shadows alias on the floor
        lights[k].data.use_shadow = False
    return floor, fm, lights


def keyed(obj, path, frames_vals, index=-1, interp="BEZIER"):
    for f, v in frames_vals:
        if index >= 0:
            getattr(obj, path)[index] = v
            obj.keyframe_insert(path, index=index, frame=f)
        else:
            setattr(obj, path, v)
            obj.keyframe_insert(path, frame=f)
    ad = obj.animation_data
    if ad and ad.action:
        for fc in ad.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = interp


def key_input(m, socket, frames_vals):
    inp = m.node_tree.nodes["Principled BSDF"].inputs[socket]
    for f, v in frames_vals:
        inp.default_value = v
        inp.keyframe_insert("default_value", frame=f)


def camera(name="cam", lens=50):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.sensor_width = 36
    cd.dof.use_dof = True
    cd.dof.aperture_fstop = 2.8
    c = bpy.data.objects.new(name, cd)
    bpy.context.collection.objects.link(c)
    bpy.context.scene.camera = c
    tgt = bpy.data.objects.new("cam_target", None)
    bpy.context.collection.objects.link(tgt)
    tc = c.constraints.new("TRACK_TO")
    tc.target, tc.track_axis, tc.up_axis = tgt, "TRACK_NEGATIVE_Z", "UP_Y"
    cd.dof.focus_object = tgt
    return c, tgt


# ── shots ────────────────────────────────────────────────────────────────────
def shot(name, bike, wheels, M, floor, fm, lights):
    c, tgt = camera()
    sc = bpy.context.scene
    on = lambda m, s, f0, f1, v: key_input(m, "Emission Strength", [(f0, 0.0), (f1, v)])  # noqa: E731

    if name == "reveal":  # darkness → DRL ignites → rim light blooms in. 3 s
        n = 90
        keyed(c, "location", [(1, (2.4, -1.5, 0.55)), (n, (1.9, -1.15, 0.62))])
        keyed(tgt, "location", [(1, (0.45, 0, 0.75)), (n, (0.35, 0, 0.7))])
        c.data.lens = 55
        for k in ("rim_l", "rim_r"):
            keyed(lights[k].data, "energy", [(1, 0), (36, 0), (66, 600)])
        keyed(lights["top"].data, "energy", [(1, 0), (50, 0), (85, 60)])
        key_input(M["drl"], "Emission Strength", [(1, 0), (22, 0), (24, 40), (26, 4), (29, 60), (34, 30)])
        key_input(M["lens"], "Emission Strength", [(1, 0), (34, 0), (38, 18)])
    elif name == "tank":  # rider's eye: over the tank, the TFT wakes up. 2.5 s
        n = 75
        c.data.lens = 45
        c.data.dof.aperture_fstop = 2.0
        keyed(c, "location", [(1, (-0.16, -0.05, 1.48)), (n, (0.08, -0.04, 1.40))])
        keyed(tgt, "location", [(1, (0.535, 0, 1.085)), (n, (0.535, 0, 1.085))])
        keyed(lights["sweep"].data, "energy", [(1, 0), (2, 220)])
        keyed(lights["sweep"], "location", [(1, (-0.6, -1.0, 1.9)), (n, (0.9, -1.0, 1.9))])
        lights["sweep"].constraints.new("TRACK_TO").target = tgt
        on(M["drl"], "", 1, 2, 30)
        key_input(M["tft"], "Emission Strength", [(1, 0), (24, 0), (26, 4), (28, 0.5), (36, 3)])
    elif name == "smg":  # engine side, SMG ring pulses on "Generator" and "push", then charges for launch. 3.5 s
        n = 105
        c.data.lens = 60
        keyed(c, "location", [(1, (0.2, 1.05, 0.38)), (n, (-0.08, 0.92, 0.46))])
        keyed(tgt, "location", [(1, (0.02, 0.1, 0.42)), (n, (0.0, 0.1, 0.42))])
        keyed(lights["key"].data, "energy", [(1, 120)])
        lights["key"].location = (1.2, 1.6, 1.2)
        lights["key"].rotation_euler = (Vector((0, 0, 0.45)) - lights["key"].location).to_track_quat("-Z", "Y").to_euler()
        key_input(M["smg"], "Emission Strength", [(1, 0), (18, 0), (21, 45), (36, 8), (57, 8), (60, 50), (74, 10), (88, 14), (105, 60)])
        on(M["drl"], "", 1, 2, 30)
    elif name == "launch":  # low tracking shot, wheels spinning, light streaks fly past. 2.5 s
        n = 75
        c.data.lens = 28
        keyed(c, "location", [(1, (1.6, -1.25, 0.28)), (n, (1.35, -1.05, 0.30))])
        keyed(tgt, "location", [(1, (0.1, 0, 0.55)), (n, (0.0, 0, 0.55))])
        for w in wheels:
            keyed(w, "rotation_euler", [(1, 0), (n, -math.tau * 6)], index=1, interp="LINEAR")
        on(M["drl"], "", 1, 2, 30)
        key_input(M["tail"], "Emission Strength", [(1, 20)])
        key_input(M["smg"], "Emission Strength", [(1, 30), (n, 12)])
        streak = mat("streak", (0, 0, 0), emit=(0.4, 0.85, 1.0), strength=25)
        for i in range(14):
            side = 1 if i % 2 else -1
            y = side * (0.9 + (i % 5) * 0.35)
            z = 0.005 if i % 3 else 0.6 + (i % 4) * 0.4
            s = box(f"streak{i}", (0, y, z), (1.6 + (i % 3), 0.02, 0.012), streak, bevel=0)
            x0 = 6 + i * 2.3
            keyed(s, "location", [(1, (x0, y, z)), (n, (x0 - 40, y, z))], interp="LINEAR")
        sc.render.use_motion_blur = True
        sc.render.motion_blur_shutter = 0.6
    elif name == "profile":  # full side profile, light blade sweeps nose to tail. 3 s
        n = 90
        c.data.lens = 35
        keyed(c, "location", [(1, (0.25, -3.6, 0.75)), (n, (-0.15, -3.5, 0.72))])
        keyed(tgt, "location", [(1, (0.05, 0, 0.62)), (n, (-0.05, 0, 0.62))])
        keyed(lights["sweep"].data, "energy", [(1, 450)])
        lights["sweep"].data.size = 0.25
        keyed(lights["sweep"], "location", [(1, (2.0, -1.6, 1.4)), (n, (-2.2, -1.6, 1.4))])
        lights["sweep"].constraints.new("TRACK_TO").target = bike
        on(M["drl"], "", 1, 2, 30)
        key_input(M["tail"], "Emission Strength", [(1, 12)])
    elif name == "hero":  # end-card hero, slow push-in, everything lit. 3 s
        n = 90
        c.data.lens = 40
        keyed(c, "location", [(1, (2.9, -2.1, 1.05)), (n, (2.45, -1.75, 0.95))])
        keyed(tgt, "location", [(1, (0.05, 0, 0.62)), (n, (0.05, 0, 0.62))])
        keyed(lights["key"].data, "energy", [(1, 160)])
        on(M["drl"], "", 1, 2, 30)
        key_input(M["smg"], "Emission Strength", [(1, 14)])
        key_input(M["tail"], "Emission Strength", [(1, 12)])
    else:
        raise SystemExit(f"unknown shot {name}")

    sc.frame_start, sc.frame_end = 1, n


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="720x1280")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--still", type=int, default=0, help="render only this frame")
    a = ap.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    bike, wheels, M = build_bike()
    if a.shot.startswith("ortho-"):  # fast geometry check: ortho workbench view
        view = a.shot[6:]
        c, tgt = camera()
        c.data.type, c.data.ortho_scale = "ORTHO", 2.2
        c.location = {"side": (0, -5, 0.6), "front": (5, 0, 0.6), "top": (0, 0, 5), "q": (3.5, -3.5, 1.6)}[view]
        if view == "q":
            c.data.ortho_scale = 2.6
        tgt.location = (0, 0, 0.6) if view != "top" else (0, 0, 0)
        if view == "top":
            c.constraints[0].up_axis = "UP_Y"
            c.location = (0, 0.001, 5)
        sc.render.engine = "BLENDER_WORKBENCH"
        sc.display.shading.light, sc.display.shading.color_type = "STUDIO", "MATERIAL"
        sc.render.resolution_x, sc.render.resolution_y = 900, 900
        sc.render.filepath = os.path.join(os.path.abspath(a.out), f"{a.shot}.png")
        bpy.ops.render.render(write_still=True)
        return
    floor, fm, lights = build_stage()
    shot(a.shot, bike, wheels, M, floor, fm, lights)

    r = sc.render
    r.engine = "BLENDER_EEVEE"
    w, h = (int(v) for v in a.size.split("x"))
    r.resolution_x, r.resolution_y, r.fps = w, h, FPS
    ee = sc.eevee
    ee.taa_render_samples = a.samples
    ee.use_bloom, ee.bloom_intensity, ee.bloom_threshold, ee.bloom_radius = True, 0.06, 0.9, 6.5
    ee.use_ssr, ee.use_ssr_refraction, ee.use_ssr_halfres = True, False, False
    ee.use_volumetric_lights, ee.volumetric_tile_size, ee.volumetric_samples, ee.volumetric_end = True, "8", 48, 14
    ee.use_gtao, ee.gtao_distance = True, 0.3
    ee.use_soft_shadows = True
    ee.shadow_cube_size, ee.shadow_cascade_size = "2048", "2048"
    ee.use_shadow_high_bitdepth = True
    sc.view_settings.exposure = -0.8
    sc.view_settings.view_transform = "AgX" if "AgX" in [i.identifier for i in sc.view_settings.bl_rna.properties["view_transform"].enum_items] else "Filmic"
    sc.view_settings.look = "AgX - Punchy" if sc.view_settings.view_transform == "AgX" else "Medium High Contrast"
    r.image_settings.file_format = "PNG"
    os.makedirs(a.out, exist_ok=True)
    r.filepath = os.path.join(os.path.abspath(a.out), "")
    if a.still:
        sc.frame_set(a.still)
        r.filepath = os.path.join(os.path.abspath(a.out), f"still_{a.still:04d}.png")
        bpy.ops.render.render(write_still=True)
    else:
        bpy.ops.render.render(animation=True)


main()
