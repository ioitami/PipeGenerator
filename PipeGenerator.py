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

class Paths:
    # TODO We'll want to able to change offset based on input
    offset = 0
    offset_radius = offset * 2 * 1.1

    vert_occupation = collections.defaultdict(int)

    vertices = None
    edges = None

    # Checking if selected_objects.size() > 1 and asking for only 1 sel obj
    sel_object = bpy.context.selected_objects[0]
    sel_object_num = len(bpy.context.selected_objects)

    if sel_object.type != 'MESH':
        print("Wrong object type!")
    if sel_object_num != 1:
        print("You can only select 1 Mesh!")

    def generate_bmesh(self, selected_obj):
        '''Create Bmesh from selected Mesh'''
        bm = bmesh.new()
        bm.from_mesh(selected_obj.data)
        return bm

    def create_mesh_to_pathfind(self, bmesh, layers=2):
        '''
            Takes in Bmesh, returns specified layers of vertices 
            and corresponding edges to use for finding paths
        '''

        vertex_layers = {}
        edge_layers = {}
        for l in range(layers):
            vertex_layers[l] = []
            edge_layers[l] = []

        vertexmap = {}
        layer_connection_edges = []
        merged_vertices = []
        all_edges = []

        for index, v in enumerate(bmesh.verts):
            # Get vertices by layers based on coordinates and normal
            
            vertex = v.co + v.normal*self.offset
            #Append vertices to each of their respective layer
            for l in range(layers):
                vertex_layers[l].append(vertex.copy())
                vertex += v.normal * self.offset_radius

            #Create vertex map index list to be used later
            vertexmap[v.index] = index

        total_vertex_num = len(vertexmap)

        for e in bmesh.edges:
            #Get index of each edge's two connected vertices
            index0 = e.verts[0].index
            index1 = e.verts[1].index
            
            #Index is called upon via the created vertexmap as it contains
            #all the vertices, then append it to the edge layer to pathfind later

            # Add edges to their respective layers
            for l in range(layers):
                edge_layers[l].append(
                    [vertexmap[index0] + l * total_vertex_num, 
                    vertexmap[index1] + l * total_vertex_num]
                    )

        #Merge all layers together
        for l in range(layers):
            merged_vertices += vertex_layers[l]
            all_edges += edge_layers[l]

        # Include edges to connect between layers
        for l in range(layers-1):
            layer_connection_edges += [[v + l * total_vertex_num, v + (l+1) * total_vertex_num] for v in range(total_vertex_num)]

        all_edges += layer_connection_edges

        #Close bmesh (Prevent further access)
        bmesh.free()
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
            faces.append({"loc": np.array(location), "orientation": n, "edges": p.loop_total})
        return faces

    def create_wires(self, start, end):
        '''
            Wires refer to skeletons of pipes
        '''
        #TODO Document this properly

        start_loc, start_n, start_e_no = start["loc"], start["orientation"], start["edges"]
        end_loc, end_n, end_e_no = end["loc"], end["orientation"], end["edges"]

        inside_start = start_loc - start_n * self.offset
        outside_start = start_loc + start_n * self.offset
        outside_end = end_loc + end_n * self.offset
        inside_end = end_loc - end_n * self.offset

        outside_verts = (mu.Vector(outside_start), mu.Vector(outside_end))

        # Check for empty neighbour vertices
        # Neighbour vertices are required as graph traversal relies on vertices and edges
        # Face values only capture the center of each face

        vert_idx, occupied = self.add_vert_edges(outside_verts, [start_e_no, end_e_no])
        if occupied:
            return vert_idx, True
        
        usable_verts, usable_edges, vert_mapping = self.get_usable_mesh()
        # for v in outside_verts:
        #     assert v in usable_verts
        outside_start_idx, outside_end_idx = vert_mapping.index(vert_idx[0]), vert_mapping.index(vert_idx[1])

        try:
            path_vertices = self.find_path(usable_verts, usable_edges, outside_start_idx, outside_end_idx, vert_mapping)
        except nx.NetworkXNoPath as ex:
            printinfo("No path found between chosen vertices")
            raise ex
        
        wire_verts = [inside_start, start_loc] + path_vertices + [end_loc, inside_end]

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
        print("vert_mapping    ",vert_mapping)
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
            length = (usable_verts[vi1] - usable_verts[vi2]).length
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

    def add_vert_edges(self, new_verts, neighbors):
        '''
            Attempts to add vertices representing the face normal at layer 0 and corresponding edges
        '''

        # Use KD-Tree to find nearest neighbouring vertices
        no_of_verts = len(self.vertices)
        kd_tree = mu.kdtree.KDTree(no_of_verts)
        for idx, v in enumerate(self.vertices):
            kd_tree.insert(v, idx)
        kd_tree.balance()

        # Identify nearest neighbouring vertices
        start_neighbors = kd_tree.find_n(new_verts[0], neighbors[0])
        end_neighbors = kd_tree.find_n(new_verts[1], neighbors[1])

        # If all neighbouring vertices are occupied, the face cannot be used
        start_occ_list = [self.vert_occupation[p[1]] for p in start_neighbors]
        start_occupied = not(0 in start_occ_list)
        end_occ_list = [self.vert_occupation[p[1]] for p in end_neighbors]
        end_occupied = not(0 in end_occ_list)

        if start_occupied or end_occupied:
            return (start_occupied, end_occupied), True
        
        new_idx = (no_of_verts, no_of_verts+1)

        self.vertices.extend(new_verts)
        self.edges.extend([[new_idx[0], p[1]] for p in start_neighbors])
        self.edges.extend([[new_idx[1], p[1]]for p in end_neighbors])
        return new_idx, False

    def render_curve(self, vert_chain, radius, res):
        '''
            Renders a curve based on a found path and other parameters
        '''
        
        # TODO Redo comments for this function (currently source's comments)


        origin = np.array((0,0,0))
        
        crv = bpy.data.curves.new('pipe', type='CURVE')

        crv.dimensions = '3D'
        crv.resolution_u = 10
        crv.bevel_depth = radius
        crv.bevel_resolution = (res - 4) // 2
        crv.extrude = 0.0 

        # make a new spline in that curve
        spline = crv.splines.new(type='POLY')

        # a spline point for each point, one point exists when initialised
        spline.points.add(len(vert_chain)-1) 
        
        # assign the point coordinates to the spline points
        node_weight = 1.0
        for p, new_co in zip(spline.points, vert_chain):
            coords = (new_co.tolist() + [node_weight])
            p.co = coords

            p.radius = 1.0 
            
        # make a new object with the curve
        obj = bpy.data.objects.new('pipe', crv)

        obj.location = origin
        return obj

    def create_pipes(self, faces, max_paths, radius, resolution, seed = 1, mat_idx = 0):
        '''
            Tries to create up to {max_paths} pipes based on faces and pathfinding area.\n
            Radius refers to pipe radius, may need to be restricted...
        '''
        #TODO Figure out resolution and mat_idx

        random.seed(seed)
        no_of_faces = len(faces)
        if no_of_faces < 2:
            msg = f"Not enough faces! Number of faces: {no_of_faces}..."
            printinfo(msg)
            return msg

        # Combination version should have an extruded version of the plane to use for pathfinding
        # Currently, the called functions use vertices, edges obtained from create_mesh_to_pathfind
        # Which are global variables
        ###PLACEHOLDER###

        ###PLACEHOLDER###

        free_faces = faces.copy()
        random.shuffle(free_faces)

        #TODO Evaluate the need for this portion
        if "pipe_collection" in self.sel_object.keys():
            if self.sel_object['pipe_collection'] in bpy.data.collections.keys():
                newcol = bpy.data.collections[self.sel_object['pipe_collection']]
            else:
                newcol = bpy.data.collections.new('pipes')
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

        for object in bpy.data.objects:
            object.select_set(False)
        self.sel_object.select_set(True)
        return "success"

# TODO Prepare to make this a plugin
# Currently the code is just run directly with the values controlled here

#bm = generate_bmesh(sel_object)
#vertices, edges = create_mesh_to_pathfind(bm, layers=2)
#faces = get_faces_from_obj_polygons(sel_object)
#msg = create_pipes(faces, max_paths = 2, radius = 0.05, resolution = 10, seed=4)













# create plugin here

    
from bpy.props import (
    BoolProperty,
    IntProperty,
    FloatProperty,
    PointerProperty
)
# class PropertyGroup(bpy.types.PropertyGroup):
#     radius: FloatProperty(name="radius",
#         description="radius of pipes",
#         min=0.001, max=1000.0,
#         step = 1.0,
#         default=.05
#     )
#     res_v: IntProperty(name="resolution v",
#         description="resolution of pipe circumference",
#         default=10,
#         min = 4
#     )
#     offset: FloatProperty(name="offset",
#         description="offset from mesh",
#         min=-10, max=10.0,
#         step = 1.0,
#         default=.11
#     )
#     seed: IntProperty(name="random seed",
#         description="seed value for randomness",
#         default=10
#     )
#     number: IntProperty(name="number of pipes",
#         description="number of pipes",
#         min = 0,
#         default = 2,
#         update=instantiate
#     )
#     layers: IntProperty(name="number of layers",
#         description="number of layers of pipes",
#         min = 0,
#         default = 2
#     )
#     reset: BoolProperty(name="reset",
#         description="delete previously created pipes",
#         default=True
#     )


#button to delete all pipes on sel_object, can be accessed in PipeGenerator panel
class delete_children(bpy.types.Operator):
    """Delete all pipes linked to this object"""
    bl_idname = "object.delete_children"
    bl_label = "Delete Object Pipes"

    def execute(self, context):
        for p_ob in context.selected_objects:
            for c_ob in p_ob.children:
                if 'pipe_id' in c_ob.keys():
                    bpy.data.objects.remove(c_ob, do_unlink=True)
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

    # radius: FloatProperty(
    #     name="radius",
    #     description="radius of pipes",
    #     min=0.001, max=1000.0,
    #     step = 1.0,
    #     default=.05
    # )
    # res_v: IntProperty(name="resolution v",
    #     description="resolution of pipe circumference",
    #     default=10,
    #     min = 4
    # )
    # offset: FloatProperty(name="offset",
    #     description="offset from mesh",
    #     min=-1000, max=1000.0,
    #     step = 1.0,
    #     default=.11
    # )
    # seed: IntProperty(name="random seed",
    #     description="seed value for randomness",
    #     default=10
    # )
    # number: IntProperty(name="number of pipes",
    #     description="number of pipes",
    #     min = 0,
    #     default = 2,
    #     update=execute
    # )
    # layers: IntProperty(name="number of layers",
    #     description="number of layers of pipes",
    #     min = 0,
    #     default = 2
    # )
    # reset: BoolProperty(name="reset",
    #     description="delete previously created pipes",
    #     default=True
    # )

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
        default = 1,
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
        
        #TODO: enable "poll" method for better object checking
        instPaths = Paths()
        instPaths.offset = self.offset
        instPaths.sel_object = bpy.context.selected_objects[0]
        
        if len(bpy.context.selected_objects) == 0:
            printinfo('No objects selected!')
            return {'CANCELLED'}
        
        if instPaths.sel_object.type != 'MESH':
            printinfo("Wrong object type!")
            return {'CANCELLED'}

        bm = instPaths.generate_bmesh(instPaths.sel_object)
        instPaths.create_mesh_to_pathfind(bm, layers=self.layers)
        # print(instPaths.vertices)
        faces = instPaths.get_faces_from_obj_polygons(instPaths.sel_object)



        state = instPaths.create_pipes(faces,
                    max_paths = self.number,
                    radius = self.radius,
                    resolution = self.res_v,
                    seed = self.seed,)        

        if state != "success":
            #self.report({'INFO'}, state)
            # Paths.render_components.catalog = object_catalog()
            # self.report({'ERROR'}, state)
            return {'CANCELLED'}
            
        return {'FINISHED'}



classes = (AddPipe, delete_children, PipePanel)



def menu_func(self, context):
    self.layout.operator(AddPipe.bl_idname, icon='META_CAPSULE')

def register():
    # bpy.utils.register_class(AddPipe)
    
    
    # bpy.utils.register_class(delete_children)
    for i in classes:
        bpy.utils.register_class(i)
    bpy.types.VIEW3D_MT_mesh_add.append(menu_func)
    # bpy.utils.register_class(IntProperty)
    # bpy.utils.register_class(PipePanel)
    # bpy.types.Scene.myProperties = PointerProperty(type=PropertyGroup)

def unregister():
    # bpy.utils.unregister_class(AddPipe)
    
    
    # bpy.utils.unregister_class(delete_children)

    for i in classes:
        bpy.utils.unregister_class(i)
    bpy.types.VIEW3D_MT_mesh_add.remove(menu_func)
    # bpy.utils.unregister_class(PropertyGroup)
    # bpy.utils.unregister_class(PipePanel)
    

if __name__ == "__main__":
    #logging.basicConfig(level=logging.INFO)     #already set in blender this way
    register()
#    unregister()
    import pdb, traceback, sys
    try:
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
