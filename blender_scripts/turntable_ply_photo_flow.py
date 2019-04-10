import bpy, os, mathutils, math
from plyfile import PlyData
import numpy as np
from os.path import join

current_dir = os.path.dirname(__file__)
input_dir= join(current_dir, "models")
output_dir = join(current_dir, "render")

sample_num = 100

for file_num, file in enumerate(os.listdir(input_dir)):
    if file.endswith(".ply"):
        file_path = os.path.join(input_dir, file)
        name = file[:-4]
        
        bpy.ops.wm.read_homefile()
        scene = bpy.data.scenes["Scene"]
        
        #parse ply (requires plyfile module inside blender)
        plydata = PlyData.read(file_path)

        #triangle size
        r = 0.01

        verts = []
        faces = []
        colors = []

        #create triangle for every vertex
        i=0
        for vertex in plydata['vertex']:
            v1 = mathutils.Vector((0, 0, r))
            v2 = mathutils.Vector((r/2, 0, 0))
            v3 = mathutils.Vector((-r/2, 0, 0))  
            
            position = mathutils.Vector((vertex['x'],vertex['y'], vertex['z']))
            normal = mathutils.Vector((vertex['nx'],vertex['ny'], vertex['nz']))
            axis = normal.cross((0,1,0))
            cosangle = normal.dot((0,1,0))
            if cosangle < -1 or cosangle > 1:
                angle = 0
            else:
                angle = math.acos(cosangle)
            rot_mat = mathutils.Matrix.Rotation(angle, 4, axis)
            
            #rotate towards normal (probably wrong)
            v1.rotate(rot_mat)
            v2.rotate(rot_mat)
            v3.rotate(rot_mat)
            
            v1 += position
            v2 += position
            v3 += position
            
            face = [3*i, 3*i+1, 3*i+2]
            color = (vertex['red']/255, vertex['green']/255, vertex['blue']/255)
            
            verts += [v1] + [v2] + [v3]
            faces += [face]
            colors += [color]
                
            i+=1

        #create mesh
        mesh = bpy.data.meshes.new("mesh_data")
        mesh.from_pydata(verts, [], faces)
        mesh.update()  

        #set vertex colors arhgbhdfgh
        mesh.vertex_colors.new()
        color_layer = mesh.vertex_colors["Col"]

        i = 0
        for poly in mesh.polygons:
            for idx in poly.vertices:
                color_layer.data[idx].color = colors[poly.index]
            i += 1

        obj = bpy.data.objects.new("My_Object", mesh)
        scene.objects.link(obj)
        
        #rotate because import reasons
        obj.rotation_euler[0] = np.pi / 2

        # settings for rendering vertex colors
        mat = bpy.data.materials.new("mat")
        mesh.materials.append(mat)
        mat.use_vertex_color_paint = True
        
        # Set render resolution
        scene.render.resolution_x = 256
        scene.render.resolution_y = 256
        scene.render.resolution_percentage = 100
        
        scene.render.filepath = output_dir
        scene.render.use_file_extension = False
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_mode = "RGBA"
        
        bpy.data.worlds["World"].horizon_color = (1,1,1)
        bpy.data.worlds["World"].light_settings.use_ambient_occlusion = True
        
        
        scene.render.layers["RenderLayer"].use_pass_vector = True
        
        scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        
        #clear default nodes
        for node in tree.nodes:
            tree.nodes.remove(node)
            
        # create input image node
        image_node = tree.nodes.new(type='CompositorNodeRLayers')
        output = tree.nodes.new('CompositorNodeComposite')   
        
        links = tree.links
        links.new(image_node.outputs["Vector"], output.inputs[0])
        
        scene.view_settings.view_transform = "Raw"
        
        
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
        cam_orig_location = mathutils.Vector((1.7, 0, 0.5))
        cam.location = cam_orig_location
        scene.camera = cam
        
        obj.parent = empty
        obj.matrix_parent_inverse = empty.matrix_world.inverted()#keep transform
        
        random = np.random.RandomState(seed=file_num)
        
        for i in range(sample_num):
            #set object orientation
            step = 2 * np.pi / sample_num
            angle = step * i
            obj.rotation_euler[2] = angle
            
            # random camera offset
            offset = mathutils.Vector(random.rand(3)).normalized() * random.rand()
            cam_offset_location = cam_orig_location + offset
            
            scene.frame_start = 0
            scene.frame_end = 1
            bpy.context.scene.frame_set(0)   
            cam.location = cam_orig_location
            cam.keyframe_insert(data_path="location")
            bpy.context.scene.frame_set(1)
            cam.location = cam_offset_location
            cam.keyframe_insert(data_path="location")

            #render from this angle
            scene.render.filepath = os.path.join(output_dir, name + "_" + str(i).zfill(3) + ".exr")
            bpy.ops.render.render( write_still=True, animation=False )   

