# blender_hand_setup.py
import bpy
import bmesh
import mathutils

def create_hand_skeleton():
    # Clear existing mesh
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Create armature (skeleton)
    bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
    armature = bpy.context.object
    armature.name = "Hand_Armature"
    
    return armature

def create_bone_hierarchy():
    """Create the complete 22-bone hierarchy for the hand"""
    bones = {}
    
    # Wrist bone (root)
    bpy.ops.armature.bone_primitive_add()
    wrist = bpy.context.active_bone
    wrist.name = "wrist"
    bones['wrist'] = wrist
    
    # Finger bones creation function
    def create_finger(finger_name, parent_bone, positions):
        finger_bones = []
        current_parent = parent_bone
        
        for i, pos in enumerate(positions):
            bpy.ops.armature.bone_primitive_add()
            bone = bpy.context.active_bone
            bone.name = f"{finger_name}_{i}"
            bone.parent = current_parent
            bone.head = current_parent.tail
            bone.tail = pos
            finger_bones.append(bone)
            current_parent = bone
            
        return finger_bones
    
    # Define finger positions (simplified - you'd adjust these)
    finger_positions = {
        'thumb': [
            mathutils.Vector((0.1, 0, 0.1)),    # CMC
            mathutils.Vector((0.2, 0, 0.2)),    # MCP  
            mathutils.Vector((0.3, 0, 0.25)),   # IP
            mathutils.Vector((0.35, 0, 0.3))    # Tip
        ],
        'index': [
            mathutils.Vector((0.1, 0.05, 0)),   # MCP
            mathutils.Vector((0.2, 0.05, 0)),   # PIP
            mathutils.Vector((0.3, 0.05, 0)),   # DIP
            mathutils.Vector((0.35, 0.05, 0))   # Tip
        ]
        # Add middle, ring, pinky similarly...
    }
    
    # Create all fingers
    bones['thumb'] = create_finger('thumb', bones['wrist'], finger_positions['thumb'])
    bones['index'] = create_finger('index', bones['wrist'], finger_positions['index'])
    # Add other fingers...
    
    return bones

# Run the setup
armature = create_hand_skeleton()
bones = create_bone_hierarchy()