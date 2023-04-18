import sys, os
import bpy
import bmesh
import random
import numpy as np
import math
import collections
from bpy_extras.object_utils import AddObjectHelper
import mathutils as mu

main_dir = os.path.join(os.path.dirname(bpy.data.filepath),'lib')
sys.path.append(main_dir)

import networkx as nx

import logging
logger = logging.getLogger(__name__)

def printinfo(msg):
    logger.info(msg)
    print(msg)
    return msg


class GeoNode_algo:
    # Geo Node algorithm

    def add_geo_nodes(node):
        if node=="GridPipe":
            GeometryNodeName = "PipeGenerator-GridFlat"
        elif node=="GridFaces":
            GeometryNodeName = "PipeGenerator-GridFaces"
        elif node == "GridWall":
            GeometryNodeName= "PipeGenerator-Wall"
        
        sel = bpy.context.active_object

        if sel is not None:
            # Duplicate Obj and rename it
            bpy.ops.object.duplicate(linked=False)
            
            new_sel = bpy.context.active_object

            new_sel.name = sel.name + " - Pipes"
            
            # 2) Add the GeometryNodes Modifier
            geo_modifier = new_sel.modifiers.new("GN", "NODES")

            # Locate the node tree you want to add to the modifier
            # Replace this with code to find the node tree you want to use
            Node_name = bpy.data.node_groups[GeometryNodeName]

            # 3) Replace the modifier's node group with the replacement
            geo_modifier.node_group = Node_name


class Paths:

    def __init__(self, offset):
        self.offset = offset
        self.offset_radius = offset * 2 * 1.1

        self.vert_occupation = collections.defaultdict(int)

        self.vertices = None
        self.edges = None

    # Checking if selected_objects.size() > 1 and asking for only 1 sel obj
    sel_object = bpy.context.selected_objects[0]
    sel_object_num = len(bpy.context.selected_objects)

    if sel_object.type != 'MESH':
        print("Wrong object type!")
    if sel_object_num != 1:
        print("You can only select 1 Mesh!")

    def create_mesh_to_pathfind(self, faces, layers=2):
        '''
            Takes in faces, returns specified layers of vertices 
            and corresponding edges to use for finding paths
        '''

        vertex_layers = {}
        edge_layers = {}
        for l in range(layers):
            vertex_layers[l] = []
            edge_layers[l] = []

        mesh_edges = []
        layer_connection_edges = []
        merged_vertices = []
        all_edges = []

        for index, v in enumerate(faces):
            # Get vertices by layers based on coordinates and normal
            
            vertex = v["loc"] + v["orientation"]*self.offset
            #Append vertices to each of their respective layer
            for l in range(layers):
                vertex_layers[l].append(vertex.copy())
                vertex += v["orientation"] * self.offset_radius

            #Remap face indices
            v["index"] = index

        total_vertex_num = len(faces)

        for f1 in faces:
            no_of_edges = 0
            for v1 in f1["edge_indices"]:
                for f2 in faces:
                    if f2["index"] == f1["index"]:
                        continue
                    no_of_edges += 1
                    if any(v2 == v1 for v2 in f2["edge_indices"]):
                        if not ([f2["index"], f1["index"]] in mesh_edges):
                            mesh_edges.append([f1["index"], f2["index"]])
            f1["edges"] = no_of_edges

        for e in mesh_edges:
            #Get index of each edge's two connected vertices
            index0 = e[0]
            index1 = e[1]
            
            #Index is called upon via the created vertexmap as it contains
            #all the vertices, then append it to the edge layer to pathfind later

            # Add edges to their respective layers
            for l in range(layers):
                edge_layers[l].append(
                    [index0 + l * total_vertex_num, 
                    index1 + l * total_vertex_num]
                    )

        #Merge all layers together
        for l in range(layers):
            merged_vertices += vertex_layers[l]
            all_edges += edge_layers[l]

        # Include edges to connect between layers
        for l in range(layers-1):
            layer_connection_edges += [[v + l * total_vertex_num, v + (l+1) * total_vertex_num] for v in range(total_vertex_num)]

        all_edges += layer_connection_edges

        replaced_edges = []
        replacing_edges = []
        extension_verts = []
        new_vert_idx = len(merged_vertices)

        for e in all_edges:
            vi1 = e[0]
            vi2 = e[1]
            vn1 = faces[vi1%total_vertex_num]["orientation"]
            vn2 = faces[vi2%total_vertex_num]["orientation"]
            if math.isclose(np.dot(np.cross(vn1, vn2), np.cross(vn1, vn2)) , 0, abs_tol=1e-9):
                continue
            v1 = merged_vertices[vi1]
            v2 = merged_vertices[vi2]
            magic_matrix = np.array([[2, 0, 0, vn1[0], vn2[0]],
                               [0, 2, 0, vn1[1], vn2[1]],
                               [0, 0, 2, vn1[2], vn2[2]],
                               [vn1[0], vn1[1], vn1[2], 0, 0],
                               [vn2[0], vn2[1], vn2[2], 0, 0]])
            magic_vec = np.array([[2*v1[0]], 
                               [2*v1[1]], 
                               [2*v1[2]], 
                               [np.dot(v1, vn1)],
                               [np.dot(v2, vn2)]])
            magic_solution = np.dot(np.linalg.inv(magic_matrix), magic_vec)
            new_vert = magic_solution[0:3].flatten().tolist()
            
            extension_verts.append(new_vert)
            replaced_edges.append(e)
            replacing_edges.append([vi1, new_vert_idx])
            replacing_edges.append([new_vert_idx, vi2])
            new_vert_idx += 1

        for r in replaced_edges:
            all_edges.remove(r)
        
        merged_vertices += extension_verts
        all_edges += replacing_edges

        self.vertices = merged_vertices
        self.edges = all_edges
        return

    def get_faces_from_obj_polygons(self, sel_obj):
        '''Takes in selected object, returns list of face locations and normals'''
        faces = []
        polygons = list(sel_obj.data.polygons)
        q_obj = sel_obj.matrix_world.to_quaternion()
        for p in polygons:
            location = p.center
            n = p.normal
            n = q_obj @ n

            # Edges here depends on loop_total (Number of loops in polygon) being equal to edges of polygon
            faces.append({"loc": np.array(location), "orientation": n, "index": p.index, "edges": p.loop_total, "edge_indices": p.edge_keys})
        return faces

    def create_wires(self, start, end):
        '''
            Wires refer to skeletons of pipes
        '''
        
        # Get the locations and indices of start and end vertices
        start_loc, start_idx = start["loc"], start["index"]
        end_loc, end_idx = end["loc"], end["index"]

        # Check for empty neighbour vertices
        occupied_verts, occupied = self.check_neighbors(start_idx, end_idx)
        if occupied:
            return occupied_verts, True
        
        usable_verts, usable_edges, vert_mapping = self.get_usable_mesh()
        assert self.vert_occupation[start_idx] == 0
        assert self.vert_occupation[end_idx] == 0

        start_idx, end_idx = vert_mapping.index(start_idx), vert_mapping.index(end_idx)

        try:
            path_vertices = self.find_path(usable_verts, usable_edges, start_idx, end_idx, vert_mapping)
        except nx.NetworkXNoPath as ex:
            printinfo("No path found between chosen vertices")
            raise ex
        
        wire_verts = [start_loc] + path_vertices + [end_loc]

        return wire_verts, False

    def get_usable_mesh(self):
        '''
            Extract only usable portions of the pathfinding mesh. \n
            Occupied vertices are considered unusable for pathfinding.
        '''
        usable_verts = []
        usable_edges = []
        vert_mapping = []
        for idx, v in enumerate(self.vertices):
            if self.vert_occupation[idx] == 0:
                usable_verts.append(v)
                vert_mapping.append(idx)
        # print("vert_mapping    ",vert_mapping)

        for e in self.edges:
            if all(self.vert_occupation[v_idx] == 0 for v_idx in e):
                usable_edges.extend([[vert_mapping.index(v_idx) for v_idx in e]])
        
        return usable_verts, usable_edges, vert_mapping

    def find_path(self, usable_verts, usable_edges, start, end, vert_mapping):
        '''
            Find path from start to end based on usable area
        '''
        graph_edges = []

        for vi1, vi2 in usable_edges:
            length = mu.Vector(usable_verts[vi1] - usable_verts[vi2]).length
            graph_edges.append([vi1, vi2, {"weight": length}])

        graph = nx.Graph(graph_edges)
        path = nx.shortest_path(graph, source=start,target=end, weight="weight")

        path_vertices = [np.array(usable_verts[vi]) for vi in path]

        self.update_vertex_occupation(path, vert_mapping)

        return path_vertices

    def update_vertex_occupation(self, path, vert_mapping):
        '''
            All vertices used in the path are marked as occupied
        '''
        for vi in path:
            self.vert_occupation[vert_mapping[vi]] += 1

    def check_neighbors(self, start, end):
        '''
            Checks that the neighbours of the start and end points are valid. \n
            If not, the points themselves cannot be used as pipe endpoints.
        '''

        # Identify nearest neighbouring vertices
        start_neighbors = []
        end_neighbors = []

        for e in self.edges:
            if e[0] == start:
                start_neighbors.append(e[1])
            elif e[1] == start:
                start_neighbors.append(e[0])
            if e[0] == end:
                end_neighbors.append(e[1])
            elif e[1] == end:
                end_neighbors.append(e[0])

        # If all neighbouring vertices are occupied, the face cannot be used
        # If the face vertex itself is occupied as well
        start_occ_list = [self.vert_occupation[p] for p in start_neighbors]
        start_occupied = not(0 in start_occ_list) or self.vert_occupation[start]
        end_occ_list = [self.vert_occupation[p] for p in end_neighbors]
        end_occupied = not(0 in end_occ_list) or self.vert_occupation[end]

        if start_occupied or end_occupied:
            return (start_occupied, end_occupied), True
        
        return (), False

    def render_curve(self, vert_chain, radius, res):
        '''
            Renders a curve based on a found path and other parameters
        '''
        
        origin = np.array((0,0,0))
        
        crv = bpy.data.curves.new('pipe', type='CURVE')

        crv.dimensions = '3D'
        crv.resolution_u = 10
        crv.bevel_depth = radius
        crv.bevel_resolution = (res - 4) // 2
        crv.extrude = 0.0 

        # Make a new spline in that curve
        spline = crv.splines.new(type='POLY')

        # Add a spline point for each point, one point exists when initialised
        spline.points.add(len(vert_chain)-1) 
        
        # Assign the point coordinates to the spline points
        node_weight = 1.0
        for p, new_co in zip(spline.points, vert_chain):
            coords = (new_co.tolist() + [node_weight])
            p.co = coords

            p.radius = 1.0 
            
        # Make a new object with the curve
        obj = bpy.data.objects.new('pipe', crv)

        obj.location = origin

        return obj

    def create_pipes(self, faces, max_paths, radius, resolution, material, seed = 1, mat_idx = 0):
        '''
            Tries to create up to {max_paths} pipes based on faces and pathfinding area.\n
            Radius refers to pipe radius, may need to be restricted...
        '''
        random.seed(seed)
        no_of_faces = len(faces)
        if no_of_faces < 2:
            msg = f"Not enough faces! Number of faces: {no_of_faces}..."
            printinfo(msg)
            return msg

        free_faces = faces.copy()
        random.shuffle(free_faces)

        if "pipe_collection" in self.sel_object.keys():
            if self.sel_object['pipe_collection'] in bpy.data.collections.keys():
                newcol = bpy.data.collections[self.sel_object['pipe_collection']]
            else:
                newname = self.sel_object.name + " - Pipes"
                newcol = bpy.data.collections.new(newname)
                bpy.context.scene.collection.children.link(newcol)
                self.sel_object['pipe_collection'] = newcol.name
        else:
            newcol = bpy.data.collections.new('pipes')
            bpy.context.scene.collection.children.link(newcol)
            self.sel_object['pipe_collection'] = newcol.name


        max_tries = 20 # Used to keep checking for possible start and end points
        curr_try = 0

        curr_paths = 0

        while (curr_paths < max_paths) and (curr_try < max_tries) and (len(free_faces) >= 2):
            try:
                wire_verts, occupied = self.create_wires(free_faces[0], free_faces[1])
                if occupied:
                    if wire_verts[0]:
                        free_faces.pop(0)
                        if wire_verts[1]:
                            free_faces.pop(0)
                    else:
                        free_faces.pop(1)
                    continue

                newobj = self.render_curve(wire_verts, radius, resolution)
                newobj["pipe_id"] = curr_paths
                newobj.parent = self.sel_object
                newcol.objects.link(newobj)

                curr_paths += 1
                free_faces = free_faces[2::]
                curr_try = 0
            except nx.NetworkXNoPath:
                # Attempt to find different configuration
                random.shuffle(free_faces)
                curr_try += 1
        if curr_try == max_tries:
            # (Likely) no more viable paths
            printinfo(f"Could not find any more paths: {curr_paths}/{max_paths} paths found")
        elif len(free_faces) < 2:
            # Creates as many paths as faces allow
            printinfo(f"No more free faces: {curr_paths}/{max_paths} paths")


        # subdivide and smooth generated pipe
        for object in bpy.data.objects:
            object.select_set(False)
        self.sel_object.select_set(False)
        
        for obj in newcol.objects:
            if ('pipe' in obj.name) and (obj.type == 'CURVE'):
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                obj.active_material = bpy.data.materials[material]
                # print("pipe added")
            else:
                obj.select_set(False)
        print(bpy.context.selected_objects)
        bpy.ops.object.mode_set(mode='EDIT', toggle=False)

        bpy.ops.curve.select_all(action='SELECT')
        bpy.ops.curve.subdivide(number_cuts=3)
        bpy.ops.curve.smooth()
        bpy.ops.curve.smooth()
        bpy.ops.curve.smooth()
        bpy.ops.curve.smooth()
        
        bpy.ops.curve.spline_type_set(type='BEZIER')
        bpy.ops.curve.decimate(ratio=0.4)
        
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

        #reset parent object back to selected state
        for object in bpy.data.objects:
            object.select_set(False)
        self.sel_object.select_set(True)
        return "success"





from bpy.props import (
    BoolProperty,
    IntProperty,
    FloatProperty,
    PointerProperty,
    EnumProperty
)

#button to delete all pipes on sel_object, can be accessed in PipeGenerator panel
class delete_children(bpy.types.Operator):
    """Delete all pipes linked to this object"""
    bl_idname = "object.delete_children"
    bl_label = "Delete Object Pipes"

    def execute(self, context):
        for p_ob in context.selected_objects:
            name = p_ob.name
            
            for c_ob in p_ob.children:
                if 'pipe_id' in c_ob.keys():
                    bpy.data.objects.remove(c_ob, do_unlink=True)
                    
        collection_name = name + " - Pipes"
        collection = bpy.data.collections.get(collection_name)
        bpy.data.collections.remove(collection)
        
        return {'FINISHED'}


#creates PipeGenerator panel on righthand tabs, shortcut = N
class PipePanel(bpy.types.Panel):
    """Creates a Panel in the scene context of the properties editor"""
    bl_label = "PipeGenerator"
    bl_idname = "SCENE_PT_piperator_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'PipeGenerator'
    #bl_context = "tool"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.operator("mesh.add_pipes") #opens Add Pipes operator
        layout.operator("object.delete_children")

        # col = self.layout.column()
        # for (prop_name, _) in PropertyGroup:
        #     row = col.row()
        #     row.prop(context.scene, prop_name)
        # row = col.row()
        # row.label(text="Number of Pipes")
        # row.prop(properties, 'number', text="")


class AddPipe(bpy.types.Operator):
    """Add pipes on faces of a mesh"""
    bl_idname = "mesh.add_pipes"
    bl_label = "Add Pipes"
    bl_options = {'REGISTER', 'UNDO'}

    # def invoke(self, context, event):
    #     self.execute(context)
    #     return {'FINISHED'}
    
    algo: EnumProperty(name="algorithm used",
        description="algorithm used to generate pipes",
        items={("py_script", "Default python script", ""),
               ("GridPipe", "GeoNode - GridFlat", ""),
               ("GridFaces", "GeoNode - GridFaces", ""),
               ("GridWall", "GeoNode - GridWall", "")},
        default="py_script"
    )
    
    material: EnumProperty(name="type of material",
        description="type of material",
        items={("Metal", "Stained Brown", ""),
               ("metal_02", "Stained Black", ""), 
               ("PipeBaseMetal2","Stained Bronze", ""),
               ("Red", "Red", "")},
        default="Metal"
    )
    
    radius: FloatProperty(name="radius",
        description="radius of pipes",
        min=0.001, max=1000.0,
        step = 1.0,
        default=.05
    )
    res_v: IntProperty(name="resolution v",
        description="resolution of pipe circumference",
        default=10,
        min = 4
    )
    offset: FloatProperty(name="offset",
        description="offset from mesh",
        min=-10, max=10.0,
        step = 1.0,
        default=.11
    )
    seed: IntProperty(name="random seed",
        description="seed value for randomness",
        default=10
    )
    number: IntProperty(name="number of pipes",
        description="number of pipes",
        min = 0,
        default = 2,
    )
    layers: IntProperty(name="number of layers",
        description="number of layers of pipes",
        min = 0,
        default = 2
    )
        
    reset: BoolProperty(name="reset",
        description="delete previously created pipes",
        default=True
    )

    
    def execute(self, context):
        if self.reset:
            for p_ob in context.selected_objects:
                for c_ob in p_ob.children:
                    if 'pipe_id' in c_ob.keys():
                        bpy.data.objects.remove(c_ob, do_unlink=True)            
        
        # add option to choose algorithm
        if self.algo=="py_script":
            #TODO: enable "poll" method for better object checking
            instPaths = Paths(self.offset)
            # instPaths.offset = self.offset
            instPaths.sel_object = bpy.context.selected_objects[0]
            
            if len(bpy.context.selected_objects) == 0:
                printinfo('No objects selected!')
                return {'CANCELLED'}
            
            if instPaths.sel_object.type != 'MESH':
                printinfo("Wrong object type!")
                return {'CANCELLED'}

            faces = instPaths.get_faces_from_obj_polygons(instPaths.sel_object)
            instPaths.create_mesh_to_pathfind(faces, layers=self.layers)
            # print(instPaths.vertices)

            state = instPaths.create_pipes(faces,
                        max_paths = self.number,
                        radius = self.radius,
                        resolution = self.res_v,
                        seed = self.seed,
                        material = self.material)        

            if state != "success":
                #self.report({'INFO'}, state)
                # Paths.render_components.catalog = object_catalog()
                # self.report({'ERROR'}, state)
                return {'CANCELLED'}
                
            return {'FINISHED'}
        
        else: 
            GeoNode_algo.add_geo_nodes(self.algo)

        
        

classes = (AddPipe, delete_children, PipePanel)

def menu_func(self, context):
    self.layout.operator(AddPipe.bl_idname, icon='META_CAPSULE')

def register():
    for i in classes:
        bpy.utils.register_class(i)
    bpy.types.VIEW3D_MT_mesh_add.append(menu_func)

def unregister():
    for i in classes:
        bpy.utils.unregister_class(i)
    bpy.types.VIEW3D_MT_mesh_add.remove(menu_func)
    

if __name__ == "__main__":
    #logging.basicConfig(level=logging.INFO)     #already set in blender this way
    import traceback, sys
    try:
        register()
        #get selected object
#        """ bm = generate_bmesh(sel_object)
#        vertices, edges = create_mesh_to_pathfind(bm, layers=layers)
#        faces = get_faces_from_obj_polygons(sel_object)
#        msg = create_pipes(faces, max_paths=number, radius=radius, resolution=res_v, seed=seed) """

#        """ob = bpy.context.selected_objects[0]
#        
#        vs, es = generate_pathfinding_mesh(ob)
#        
#        newobj = gu.genobjfrompydata(verts = vs,
#                                 edges = es)"""
        # test call
        # bpy.ops.mesh.add_pipes()
        pass
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        #pdb.post_mortem(tb)