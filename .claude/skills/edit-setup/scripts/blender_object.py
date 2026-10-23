"""Headless Blender: render a 3D object as a transparent turntable for HyperFrames.

  blender -b -P blender_object.py -- --out assets/obj [--model thing.glb]
      [--primitive cube|sphere|torus|monkey] [--color "#D9653B"]
      [--frames 90] [--fps 30] [--size 1080] [--engine auto|EEVEE|CYCLES]

Writes <out>/####.png (RGBA). pipeline.sh then encodes them to a VP9-alpha
.webm that a HyperFrames <video> clip can layer over footage — zero credits.
"""

import argparse
import math
import os
import sys

import bpy


def parse() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model")
    ap.add_argument("--primitive", default="cube")
    ap.add_argument("--color", default="#D9653B")
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", type=int, default=1080)
    ap.add_argument("--engine", default="auto")
    return ap.parse_args(argv)


def hex_rgba(h: str) -> tuple:
    h = h.lstrip("#")
    srgb = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    return (*lin, 1.0)


def main() -> None:
    a = parse()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    if a.model:
        ext = os.path.splitext(a.model)[1].lower()
        if ext in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=a.model)
        elif ext == ".obj":
            bpy.ops.wm.obj_import(filepath=a.model)
        elif ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=a.model)
        else:
            raise SystemExit(f"unsupported model type: {ext}")
        meshes = [o for o in scene.objects if o.type == "MESH"]
        bpy.ops.object.select_all(action="DESELECT")
        for o in meshes:
            o.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active
    else:
        add = {
            "cube": lambda: bpy.ops.mesh.primitive_cube_add(size=1.4),
            "sphere": lambda: bpy.ops.mesh.primitive_uv_sphere_add(radius=0.9, segments=64, ring_count=32),
            "torus": lambda: bpy.ops.mesh.primitive_torus_add(major_radius=0.8, minor_radius=0.3),
            "monkey": lambda: bpy.ops.mesh.primitive_monkey_add(size=1.6),
        }[a.primitive]
        add()
        obj = bpy.context.active_object
        bpy.ops.object.shade_smooth()
        bev = obj.modifiers.new("bevel", "BEVEL")
        bev.width, bev.segments = 0.06, 4
        mat = bpy.data.materials.new("hf")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = hex_rgba(a.color)
        bsdf.inputs["Roughness"].default_value = 0.35
        obj.data.materials.append(mat)

    # Normalize: center at origin, fit in a 2-unit box.
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0, 0, 0)
    s = 2.0 / max(obj.dimensions)
    obj.scale = (s, s, s)

    # Three-point light + camera.
    for name, loc, energy in (("key", (4, -4, 5), 1200), ("fill", (-5, -2, 2), 400), ("rim", (0, 5, 4), 800)):
        light = bpy.data.lights.new(name, "AREA")
        light.energy, light.size = energy, 4
        lo = bpy.data.objects.new(name, light)
        lo.location = loc
        scene.collection.objects.link(lo)
        lo.rotation_euler = (lo.location * -1).to_track_quat("-Z", "Y").to_euler()
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.location = (0, -6, 1.6)
    cam.rotation_euler = (math.radians(76), 0, 0)
    scene.collection.objects.link(cam)
    scene.camera = cam

    # Turntable: one full revolution, linear so it loops seamlessly.
    obj.rotation_euler = (0, 0, 0)
    obj.keyframe_insert("rotation_euler", index=2, frame=1)
    obj.rotation_euler[2] = math.tau
    obj.keyframe_insert("rotation_euler", index=2, frame=a.frames + 1)
    for fc in obj.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"

    r = scene.render
    engine = a.engine.upper()
    if engine == "AUTO":
        # EEVEE needs a GPU/EGL context; headless Linux servers only have Cycles on CPU.
        headless = sys.platform.startswith("linux") and not os.environ.get("DISPLAY")
        engine = "CYCLES" if headless else "EEVEE"
    engines = {e.identifier for e in r.bl_rna.properties["engine"].enum_items}
    if engine == "EEVEE":
        r.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    else:
        r.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 64
        # Distro builds often ship without OpenImageDenoise; denoise only when available.
        scene.cycles.use_denoising = getattr(bpy.app.build_options, "openimagedenoise", False)
    r.resolution_x = r.resolution_y = a.size
    r.fps = a.fps
    r.film_transparent = True
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGBA"
    scene.frame_start, scene.frame_end = 1, a.frames
    os.makedirs(a.out, exist_ok=True)
    r.filepath = os.path.join(os.path.abspath(a.out), "")
    bpy.ops.render.render(animation=True)


main()
