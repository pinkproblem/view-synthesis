import bpy, os, mathutils, math
import numpy as np
from os.path import join

current_dir = os.path.dirname(__file__)
input_dir= join(current_dir, "models")
output_dir = join(current_dir, "render")

sample_num = 100

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
        #bpy.data.worlds["World"].ambient_color = (0.1, 0.1, 0.1)
        bpy.data.worlds["World"].light_settings.use_ambient_occlusion = True
        #bpy.data.worlds["World"].light_settings.use_environment_light = True
        
        
        scene.render.use_freestyle = True
        scene.render.layers["RenderLayer"].use_solid = False
        linestyle = bpy.data.linestyles["LineStyle"]
        linestyle.thickness = 1.0
        
        
        cam = bpy.data.objects['Camera']
        #add empty for camera orientation
        bpy.ops.object.empty_add(type='PLAIN_AXES',radius=1,location=(0,0,0.4))
        empty = bpy.context.active_object
        #track to empty
        track = cam.constraints.new(type='TRACK_TO')
        track.target = empty
        track.track_axis = 'TRACK_NEGATIVE_Z'
        track.up_axis = 'UP_Y'
        
        # turntable
        cam.location = (1.7, 0, 0.5)
        scene.camera = cam

        for i in range(0, sample_num):
            step = 2 * np.pi / sample_num
            angle = step * i
            obj.rotation_euler[2] = angle  
  
            #render from this angle
            scene.render.filepath = os.path.join(output_dir, name + '_' + str(i).zfill(3))
            bpy.ops.render.render( write_still=True )   

