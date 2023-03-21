import bpy
from math import sqrt

curve = bpy.context.object


if len(curve.data.splines) > 1:

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.curve.select_all(action='DESELECT')

radius = 0.25
curves = [i for i in curve.data.splines]

for c in curves:
    for p in c.points:
        for check in curves:
            if check == c:
                continue
            for point in check.points:
                dist = sqrt( (point.co[0]-p.co[0])**2 + (point.co[1]-p.co[1])**2 + (point.co[2]-p.co[2])**2)
                if dist < 2*radius:
                    p.co = [p.co[0], p.co[1], p.co[2]+2.5*radius, p.co[3]]

while len(curve.data.splines) > 1:

    for point in curve.data.splines[0].points:
        point.select = True
        
    bpy.ops.curve.separate()
                    

bpy.ops.object.mode_set(mode='OBJECT')
