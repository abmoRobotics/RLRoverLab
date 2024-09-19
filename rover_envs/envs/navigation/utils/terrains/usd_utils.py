import numpy as np
import omni.isaac.core.utils.prims as prim_utils
# import omni.isaac.lab.utils.kit as kit_utils
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.stage import get_current_stage
from pxr import PhysxSchema, Usd, UsdPhysics


def trimesh_to_usd(vertices: np.ndarray, faces: np.ndarray, position=None, orientation=None, name="terrain"):
    """ Convert trimesh to USD

    Parameters
    ----------
    vertices : np.ndarray
        Vertices of the mesh
    faces : np.ndarray
        Faces of the mesh
    """

    # Get current stage
    stage: Usd.Stage = get_current_stage()

    # Define terrain mesh
    mesh_prim = stage.DefinePrim(f"/World/{name}", "Mesh")
    mesh_prim.GetAttribute("points").Set(vertices)
    mesh_prim.GetAttribute("faceVertexIndices").Set(faces.flatten())
    mesh_prim.GetAttribute("faceVertexCounts").Set(np.asarray([3] * faces.shape[0]))  # 3 vertices per face

    terrain_prim = XFormPrim(
        prim_path=f"/World/{name}",
        name=f'{name}',
        position=position,
        orientation=orientation)

    UsdPhysics.CollisionAPI.Apply(terrain_prim.prim)

    physx_collision_api: PhysxSchema._physxSchema.PhysxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(
        terrain_prim.prim)

    physx_collision_api.GetContactOffsetAttr().Set(0.04)
    physx_collision_api.GetRestOffsetAttr().Set(0.01)

    material = PhysicsMaterial(
        prim_path=f"/World/Materials/{name}",
        # static_friction=0.1,
        # dynamic_friction=0.8,
        static_friction=0.1,
        dynamic_friction=2,
        restitution=0.0,
    )
    material2: PhysxSchema._physxSchema.PhysxMaterialAPI = PhysxSchema.PhysxMaterialAPI.Apply(material.prim)
    material2.CreateCompliantContactStiffnessAttr().Set(1000000.0)
    material2.CreateCompliantContactDampingAttr().Set(20000.0)

    # PhysxSchema._physxSchema.PhysxMaterialAPI.CreateCompliantContactStiffnessAttr(defaultValue=0.0,
    # writeSparsely=False)
    # kit_utils.apply_nested_physics_material(terrain_prim.prim_path, material.prim_path)

# TODO: Clean up this function and potentially remove it


def add_material_to_stage_from_mdl():
    # stage: Usd.Stage = get_current_stage()

    import omni.kit.commands

    omni.kit.commands.execute('CreateAndBindMdlMaterialFromLibrary',
                              mdl_name='omniverse://localhost/NVIDIA/Materials/Base/Natural/Soil_Rocky.mdl',
                              mtl_name='Soil_Rocky',
                              mtl_created_list=['/Looks/Soil_Rocky'],
                              select_new_prim=False)

    omni.kit.commands.execute('CreateAndBindMdlMaterialFromLibrary',
                              mdl_name='omniverse://localhost/NVIDIA/Materials/Base/Stone/Fieldstone.mdl',
                              mtl_name='Fieldstone',
                              mtl_created_list=['/Looks/Fieldstone'],
                              select_new_prim=False)

    # import time
    # time.sleep(3)
    # omni.kit.commands.execute('ChangeProperty',
    #     prop_path=Sdf.Path('/Looks/Soil_Rocky/Shader.inputs:project_uvw'),
    #     value=True,
    #     prev=None)

    # # omni.kit.commands.execute('ChangeProperty',
    # #     prop_path=Sdf.Path('/Looks/Fieldstone/Shader.inputs:flip_tangent_u'),
    # #     value=True,
    # #     prev=None)

    # omni.kit.commands.execute('SelectPrims',
    #         old_selected_paths=['/Looks'],
    #         new_selected_paths=['/Looks/Soil_Rocky'],
    #         expand_in_stage=True)

    prim_utils.create_prim(
        "/World/AmbientLight",
        "DistantLight",
        attributes={"inputs:intensity": 600.0},  # Isaac Sim 2023
        # attributes={"intensity": 600.0},  # Isaac Sim 2022
    )

    # omni.kit.commands.execute(
    #         "ChangeProperty",
    #         prop_path=f"World/SphereLight.xformOp:translate",
    #         value=(0.0, 0.0, 1.5 * light_radius),
    #         prev=None,
    #     )
    # soil_rocky_prim = stage.GetPrimAtPath('/Looks/Soil_Rocky/Shader')
    # print(f'soil_rocky_prim: {soil_rocky_prim.GetAttributes("inputs:project_uvw").Get()}')
    # soil_rocky_prim.GetAttribute("inputs:project_uvw").Set(True)
    # soil_rocky_prim.GetAttribute("inputs:texture_scale").Set((0.3,0.3))

    # # soil_rocky_prim.getpro

    # # omni.kit.commands.execute('ChangeProperty',
    # #     prop_path=Sdf.Path('/Looks/Soil_Rocky/Shader.inputs:texture_scale'),
    # #     value=Gf.Vec2f(0.1, 0.1),
    # #     prev=None)
    # soil_rocky_prim = stage.GetPrimAtPath('/Looks/Soil_Rocky/Shader')
    # soil_rocky_prim2 = stage.GetPrimAtPath('/Looks/Soil_Rocky')
    # print(f'soil_rocky_prim: {soil_rocky_prim}')
    # print(f'soil_rocky_prim2: {soil_rocky_prim2}')
    # print(f'soil_rocky_prim2.GetAttribute("inputs:project_uvw"): {soil_rocky_prim.GetAttributes()}')
    # try:
    #     soil_rocky_prim.GetAttribute("inputs:project_uvw").Set(True)
    # except Exception as e:
    #     print(f'Exception: {e}')

    # fieldstone_prim.GetAttribute("inputs:project_uvw").Set(True)
    # /Looks/Soil_Rocky/Shader.inputs:project_uvw
    # Type float2
    # soil_rocky_prim.GetAttribute("inputs:texture_scale").Set([1.0, 1.0])


def get_triangles_and_vertices_from_prim(prim_path):
    """ Get triangles and vertices from prim """
    stage: Usd.Stage = get_current_stage()
    mesh_prim = stage.GetPrimAtPath(prim_path)

    points = mesh_prim.GetAttribute("points").Get()
    # face_vertex_counts = mesh_prim.GetAttribute("faceVertexCounts").Get()
    face_vertex_indices = mesh_prim.GetAttribute("faceVertexIndices").Get()

    vertices = [(point[0], point[1], point[2]) for point in points]
    faces = [(face_vertex_indices[i], face_vertex_indices[i+1], face_vertex_indices[i+2])
             for i in range(0, len(face_vertex_indices), 3)]

    return faces, vertices


def apply_material(prim_path, material_path="/Looks/Soil_Rocky"):
    """ Apply material to prim """
    import omni.kit.commands

    omni.kit.commands.execute('BindMaterial',
                              material_path=material_path,
                              prim_path=[prim_path],
                              strength=['weakerThanDescendants'])

    # stage: Usd.Stage = get_current_stage()
    # prim = stage.GetPrimAtPath(prim_path)

    # Apply material
