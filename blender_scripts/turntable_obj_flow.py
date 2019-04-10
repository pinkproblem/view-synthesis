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
        
        #scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
        #scene.render.image_settings.color_mode = "RGBA"
        
        obj = bpy.context.selected_objects[0]
        
        # Set render resolution
        scene.render.resolution_x = 400
        scene.render.resolution_y = 300
        scene.render.resolution_percentage = 100
        
        bpy.data.worlds["World"].horizon_color = (1,1,1)
        #bpy.data.worlds["World"].ambient_color = (0.1, 0.1, 0.1)
        bpy.data.worlds["World"].light_settings.use_ambient_occlusion = True
        bpy.data.worlds["World"].color_range = 3
        #bpy.data.worlds["World"].light_settings.use_environment_light = True
        

        scene.render.layers["RenderLayer"].use_pass_vector = True
        
        scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        
        #clear default nodes
        for node in tree.nodes:
            tree.nodes.remove(node)
            
        # create input image node
        image_node = tree.nodes.new(type='CompositorNodeRLayers')
        output = tree.nodes.new('CompositorNodeOutputFile')
        output.base_path = output_dir
        output.file_slots[0].path = name + "_###"
        output.format.file_format = 'OPEN_EXR'
        
        links = tree.links
        links.new(image_node.outputs["Vector"], output.inputs[0])
        #links.new(image_node.outputs[0], comp_node.inputs[0])
        
        scene.view_settings.view_transform = "Raw"
        
        
        cam = bpy.data.objects['Camera']
        #add empty for camera orientation
        bpy.ops.object.empty_add(type='PLAIN_AXES',radius=1,location=(0,0,0.5))
        empty = bpy.context.active_object
        #track to empty
        track = cam.constraints.new(type='TRACK_TO')
        track.target = empty
        track.track_axis = 'TRACK_NEGATIVE_Z'
        track.up_axis = 'UP_Y'
        
        # turntable
        cam.location = (2, 0, 0.5)
        scene.camera = cam
        
        obj.parent = empty
        obj.matrix_parent_inverse = empty.matrix_world.inverted()#keep transform
        
        scene.frame_start = 0
        scene.frame_end = sample_num - 1
        bpy.context.scene.frame_set(0)   
        empty.rotation_euler[2] = 0
        empty.keyframe_insert(data_path="rotation_euler")
        bpy.context.scene.frame_set(sample_num)
        empty.rotation_euler[2] = math.radians(360)
        empty.keyframe_insert(data_path="rotation_euler")
        
        #set curves to linear
        for fc in empty.animation_data.action.fcurves:
            fc.extrapolation = 'LINEAR'
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR' 

        #render from this angle
        #scene.render.filepath = os.path.join(output_dir, name)
        bpy.ops.render.render( write_still=False, animation=True )   

