import bpy, os, mathutils, math
import numpy as np
from os.path import join

current_dir = os.path.dirname(__file__)
input_dir= join(current_dir, "models")
output_dir = join(current_dir, "render")

sample_num = 200

for file in os.listdir(input_dir):
    if file.endswith(".obj"):
        file_path = os.path.join(input_dir, file)
        name = file[:-4]
             
        bpy.ops.wm.read_homefile()
        bpy.ops.import_scene.obj(filepath = file_path)
        scene = bpy.data.scenes["Scene"]
        
        obj = bpy.context.selected_objects[0]
        
        # Set render resolution
        scene.render.resolution_x = 256
        scene.render.resolution_y = 256
        scene.render.resolution_percentage = 100
        
        bpy.data.worlds["World"].horizon_color = (1,1,1)
        
        
        scene.render.use_freestyle = True
        scene.render.layers["RenderLayer"].use_solid = False
        linestyle = bpy.data.linestyles["LineStyle"]
        
        linestyle.chaining = "SKETCHY"
        
        bpy.ops.scene.freestyle_thickness_modifier_add(type='ALONG_STROKE')
        linestyle.thickness_modifiers["Along Stroke"].value_min = 1.
        linestyle.thickness_modifiers["Along Stroke"].value_max = 2.
        
        #alpha
        bpy.ops.scene.freestyle_alpha_modifier_add(type='ALONG_STROKE')
        linestyle.alpha_modifiers["Along Stroke"].influence = 0.5

        #geometry
        bpy.ops.scene.freestyle_geometry_modifier_add(type='SPATIAL_NOISE')
        linestyle.geometry_modifiers["Spatial Noise"].amplitude = 3.
        linestyle.geometry_modifiers["Spatial Noise"].scale = 100.
        linestyle.geometry_modifiers["Spatial Noise"].octaves = 2
        
        # texture       
        linestyle.use_texture = True
        tex = bpy.data.textures.new("texture", 'IMAGE')
        img = bpy.data.images.load(join(current_dir, "charcoal.png"))
        img.use_alpha = True
        tex.image = img
        tex.use_alpha = True
        linestyle.active_texture = tex
        linestyle.texture_slots[0].blend_type = "MULTIPLY"
        linestyle.texture_slots[0].use_map_color_diffuse = False
        linestyle.texture_slots[0].use_map_alpha = True
        linestyle.texture_slots[0].alpha_factor = 1.
        
        tex.use_calculate_alpha = True
        tex.filter_eccentricity = 32.
        tex.filter_size = 0.1
        
        
        cam = bpy.data.objects['Camera']
        #add empty for camera orientation
        bpy.ops.object.empty_add(type='PLAIN_AXES',radius=1,location=(0,0,0.4))
        empty = bpy.context.active_object
        #track to empty
        track = cam.constraints.new(type='TRACK_TO')
        track.target = empty
        track.track_axis = 'TRACK_NEGATIVE_Z'
        track.up_axis = 'UP_Y'

        r = 1.7
        for i in range(0, sample_num):
            np.random.seed(i)
            theta = np.random.rand() * np.pi
            a = np.random.rand() * 2 - 1
            phi = np.arccos(a)
            x = r * np.cos(theta) * np.sin(phi)
            z = r * np.sin(theta) * np.sin(phi)
            y = r * np.cos(phi)
            
            cam.location = (x, y, z)
            scene.camera = cam
  
            #render from this angle
            scene.render.filepath = os.path.join(output_dir, name + '_' + str(i).zfill(3))
            bpy.ops.render.render( write_still=True )   

