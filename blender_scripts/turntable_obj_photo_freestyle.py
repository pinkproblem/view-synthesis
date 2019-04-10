import bpy, os, mathutils, math
import numpy as np
from os.path import join

current_dir = os.path.dirname(__file__)
input_dir= join(current_dir, "models")
output_dir = join(current_dir, "render")

sample_num = 100

for file_num, file in enumerate(os.listdir(input_dir)):
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
        bpy.data.worlds["World"].light_settings.use_ambient_occlusion = True
        
        scene.view_settings.view_transform = "Raw"
        
        #set up uniform light

        #remove all lamps
        for l in bpy.data.objects:
            if l.type == "LAMP":
                bpy.data.objects.remove(l, do_unlink=True)
        #add a new one
        lamp_data = bpy.data.lamps.new(name="New Lamp", type='POINT')
        lamp_object = bpy.data.objects.new(name="New Lamp", object_data=lamp_data)
        scene.objects.link(lamp_object)
        lamp_object.location = (5.0, 5.0, 5.0)
        
        
        cam = bpy.data.objects['Camera']
        #add empty for camera orientation
        bpy.ops.object.empty_add(type='PLAIN_AXES',radius=1,location=(0,0,0.4))
        empty = bpy.context.active_object
        #track to empty
        track = cam.constraints.new(type='TRACK_TO')
        track.target = empty
        track.track_axis = 'TRACK_NEGATIVE_Z'
        track.up_axis = 'UP_Y'
        
        
        scene.render.use_freestyle = True
        scene.render.layers["RenderLayer"].use_solid = False
        linestyle = bpy.data.linestyles["LineStyle"]
        linestyle.thickness = 1.0
        
        # turntable
        cam_orig_location = mathutils.Vector((1.7, 0, 0.5))
        cam.location = cam_orig_location
        scene.camera = cam
        
        random = np.random.RandomState(seed=file_num)

        for i in range(0, sample_num):
            step = 2 * np.pi / sample_num
            angle = step * i
            obj.rotation_euler[2] = angle  
            
            # random camera offset
            offset = mathutils.Vector(random.rand(3)).normalized() * random.rand()
            cam_offset_location = cam_orig_location + offset
            cam.location = cam_offset_location
  
            #render from this angle
            scene.render.filepath = os.path.join(output_dir, name + '_' + str(i).zfill(3))
            bpy.ops.render.render( write_still=True )   

