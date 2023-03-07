from ast import Index
from multiprocessing.sharedctypes import Value
from geometry_script import *

@tree("Extrude Index Test")
def Extrude_Index(Mesh: Geometry, exact_ID: Int, ExtrudeX: Float, ExtrudeY: Float, ExtrudeZ: Float ):

    cmpr_id = math(operation=Math.Operation.COMPARE, value=(index(), exact_ID, 0))

    ex = extrude_mesh(
        mesh=Mesh, selection=cmpr_id, 
        offset=combine_xyz(x=ExtrudeX,y=ExtrudeY, z=ExtrudeZ)
    )

    return {"Mesh":ex.mesh}