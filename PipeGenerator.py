import sys, os
import bpy
import bmesh
import random
import numpy as np
import math
import collections
from bpy_extras.object_utils import AddObjectHelper
import mathutils as mu
import networkx as nx

import importlib
import helpers.mathhelp as mh
import helpers.genutils as gu
import helpers.generationgraph as gg
importlib.reload(mh)
importlib.reload(gu)
importlib.reload(gg)


# We'll want to able to change offset based on input
offset= 0.1
offset_radius = offset * 2 * 1.1

# Checking if selected_objects.size() > 1 and asking for only 1 sel obj
sel_object = bpy.context.selected_objects[0]
sel_object_num = len(bpy.context.selected_objects)

if sel_object.type != 'MESH':
    print("Wrong object type!")
if sel_object_num != 1:
    print("You can only select 1 Mesh!")

def generate_bmesh(selected_obj):
    #Create Bmesh from selected Mesh
    bm = bmesh.new()
    bm.from_mesh(selected_obj.data)
    return bm

def create_mesh_to_pathfind(bmesh):
    vertexlayer1 = []
    vertexlayer2 = []
    edgelayer = []
    edgelayer2 = []
    edge_connections = []
    all_edges = []

    vertexmap = {}

    for index, v in enumerate(bmesh.verts):
        #Get vertex via their normal with...
        vertex1 = v.co + v.normal * offset
        vertex2 = v.co + v.normal * (offset + offset_radius)

        #Append vertices to each of their respective layer
        vertexlayer1.append(vertex1)
        vertexlayer2.append(vertex2)

        #Create vertex map index list to be used later
        vertexmap[v.index] = index

    for e in bmesh.edges:
        #Get index of each Edge's two connected vertices
        index0 = e.verts[0].index
        index1 = e.verts[1].index
        
        #Index is called upon via the created vertexmap as it contains
        #all the vertices, then append it to the edge layer to pathfind later
        edgelayer.append([vertexmap[index0], vertexmap[index1]])

    #Merge the two layers together
    total_vertex_num = len(vertexmap)
    merged_vertices = vertexlayer1 + vertexlayer2

    edgelayer2 = [[v+total_vertex_num for v in edge] for edge in edgelayer]
    layer_connection_edges = [[v, v+total_vertex_num] for v in range(total_vertex_num)]
    #Combine all edges to make the pathfinding mesh
    all_edges = edgelayer + edgelayer2 + layer_connection_edges

    #Close bmesh (Prevent further access)
    bmesh.free()
    return merged_vertices, all_edges

def get_interface_from_obj_polygons():
    return

bm = generate_bmesh(sel_object)
vertes_edges = create_mesh_to_pathfind(bm)