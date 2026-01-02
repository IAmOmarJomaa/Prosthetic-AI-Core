# blender_hand_mesh.py
def create_hand_mesh():
    """Create the hand geometry and skin it to the armature"""
    
    # Create basic hand shape using metaballs or sculpting
    bpy.ops.object.metaball_add(type='BALL', location=(0, 0, 0))
    hand_mesh = bpy.context.object
    hand_mesh.name = "Hand_Mesh"
    
    # Add subsurface modifier for smoothness
    hand_mesh.modifiers.new("Subdivision", type='SUBSURF')
    hand_mesh.modifiers["Subdivision"].levels = 2
    
    return hand_mesh

def rig_hand_to_armature(hand_mesh, armature):
    """Skin the hand mesh to the armature"""
    
    # Add armature modifier
    armature_modifier = hand_mesh.modifiers.new("Armature", type='ARMATURE')
    armature_modifier.object = armature
    
    # Enter weight painting mode (you'd need to paint weights properly)
    bpy.context.view_layer.objects.active = hand_mesh
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    
    return hand_mesh

# Complete the setup
hand_mesh = create_hand_mesh()
rigged_hand = rig_hand_to_armature(hand_mesh, armature)