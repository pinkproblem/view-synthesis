import bpy, os, mathutils, math
import numpy as np
from plyfile import PlyData
from os.path import join

current_dir = os.path.dirname(__file__)
input_dir= join(current_dir, "models")
output_dir = join(current_dir, "render")

sample_num = 200

for file in os.listdir(input_dir):
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
        
        bpy.data.worlds["World"].horizon_color = (1,1,1)
        bpy.data.worlds["World"].light_settings.use_ambient_occlusion = True
        scene.view_settings.view_transform = "Raw"
        
        
        cam = bpy.data.objects['Camera']
        #add empty for camera orientation
        bpy.ops.object.empty_add(type='PLAIN_AXES',radius=1,location=(0.,0.,0.4))
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

            
           
            
            
            
            
            

