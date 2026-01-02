# blender_export.py
def export_hand_model():
    """Export the rigged hand model to GLTF format"""
    
    bpy.ops.export_scene.gltf(
        filepath="C:/path/to/your/hand_model.gltf",  # Update this path
        export_format='GLTF_EMBEDDED',
        export_yup=True,
        export_apply=True,
        export_animations=False,
        export_skins=True,
        export_morph=False
    )

export_hand_model()