from disney import DisneyMaterial
from luisa.mathtypes import *

def create_mat(**kwargs):
    mat = DisneyMaterial.default.copy()
    for name in kwargs:
        assert name in mat.structType.idx_dict
        setattr(mat, name, kwargs[name])
    return mat

LeftWall = create_mat(base_color=float3(0.63, 0.065, 0.05))
RightWall = create_mat(base_color=float3(0.14, 0.45, 0.091))
GrayMatte = create_mat(base_color=float3(0.725, 0.71, 0.68), roughness=0.2)
Light = create_mat(base_color=float3(0))
mat_Glass = create_mat(base_color=float3(1.0), roughness=0.1, ior=1.5, specular_transmission=1.0)


models = [
("cbox_disney/models/floor.obj", GrayMatte),
("cbox_disney/models/ceiling.obj", GrayMatte),
("cbox_disney/models/backwall.obj", GrayMatte),
("cbox_disney/models/rightwall.obj", RightWall),
("cbox_disney/models/leftwall.obj", LeftWall),
("cbox_disney/models/shortblock.obj", mat_Glass),
("cbox_disney/models/tallblock.obj", GrayMatte),
("cbox_disney/models/light.obj", Light, float3(17,12,4))
]

const_env_light = float3(0.0)

camera_pos = float3(278, 273, -800)
camera_dir = float3(0,0,1)
camera_up = float3(0.0, 1.0, 0.0)
camera_fov = 39.0

resolution = 1024, 1024
max_depth = 8
rr_depth = 3