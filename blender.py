import os
import bpy


working_dir_path = os.path.dirname(os.path.abspath(__file__))

def import_trajectory(name, name_mat0):
    bpy.ops.import_scene.obj(filepath=working_dir_path+"/result/{}.obj".format(name))
    obj = bpy.data.objects[name]
    obj.scale = (0.01, 0.01, 0.01)
    bpy.context.scene.view_layers[0].objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.convert(target='CURVE')
    bpy.data.curves[name].bevel_depth = 0.3
    obj.data.materials.append(bpy.data.materials[name_mat0])
    # obj.material_slots[0].material = bpy.data.materials[name_mat0]


def import_character(name, name_mat0):
    bpy.ops.import_scene.obj(filepath=working_dir_path+"/result/{}.obj".format(name))
    obj = bpy.data.objects[name]
    obj.scale = (0.01, 0.01, 0.01)
    bpy.context.scene.view_layers[0].objects.active = obj
    bpy.ops.object.modifier_add(type='WIREFRAME')
    obj.modifiers['Wireframe'].thickness = 0.4
    obj.modifiers['Wireframe'].use_replace = False
    obj.modifiers['Wireframe'].material_offset = 1
    obj.material_slots[0].material = bpy.data.materials[name_mat0]
    obj.data.materials.append(bpy.data.materials['Black'])


if __name__ == "__main__":

    # remove cube
    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete(use_global=False)

    # light pos
    obj = bpy.data.objects['Light']
    obj.location = (0,0,5)

    # materials
    material_black = bpy.data.materials.new('Black')
    material_black.use_nodes = False
    material_black.diffuse_color = (0, 0, 0, 1)
    material_black.roughness = 0.0
    material_black.specular_intensity = 0.0

    material_floor = bpy.data.materials.new('Floor')
    material_floor.use_nodes = True

    material_red = bpy.data.materials.new('Red')
    material_red.use_nodes = False
    material_red.diffuse_color = (1., 0.2, 0.2, 1)

    material_blue = bpy.data.materials.new('Blue')
    material_blue.use_nodes = False
    material_blue.diffuse_color = (0.1, 0.2, 1.0, 1)

    # floor
    bpy.ops.mesh.primitive_plane_add(size=200)


    name = '138_02'
    import_trajectory('{}_traj'.format(name), 'Red')    
    for i in [0,100,200,300,400,500,600,700]:
        import_character( '{}_frame_{}'.format(name,i),'Red')

