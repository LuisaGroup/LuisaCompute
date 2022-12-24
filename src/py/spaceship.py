from disney import DisneyMaterial
from luisa import lcapi
from luisa.mathtypes import *

def create_mat(**kwargs):
    mat = DisneyMaterial.default.copy()
    for name in kwargs:
        assert name in mat.structType.idx_dict
        setattr(mat, name, kwargs[name])
    return mat


mat_Backdrop = create_mat(base_color=float3(0.1), roughness=1.0)
mat_RoughAluminium = create_mat(base_color=float3(1.0), roughness=0.316227766, metallic=1.0)
mat_RoughSteel = create_mat(base_color=float3(0.1), roughness=0.316227766, metallic=1.0)
mat_DarkPlastic = create_mat(base_color=float3(0.01), roughness=0.4472135955, ior=1.5)
mat_PinkLeather = create_mat(base_color=float3(0.256, 0.013, 0.08), roughness=0.632455532, ior=1.5)
mat_Leather = create_mat(base_color=float3(0.034, 0.014, 0.008), roughness=0.632455532, ior=1.5)
mat_RedLeather = create_mat(base_color=float3(0.163, 0.03, 0.037), roughness=0.632455532, ior=1.5)
mat_BrightPinkLeather = create_mat(base_color=float3(0.772, 0.175, 0.262), roughness=0.632455532, ior=1.5)
mat_Glass = create_mat(base_color=float3(1.0), roughness=0.1, ior=1.5, specular_transmission=1.0)
mat_DarkRubber = create_mat(base_color=float3(0.025), roughness=0.632455532, ior=1.5)
mat_Black = create_mat(base_color=float3(0.0))
mat_Null = create_mat(base_color=float3(0.0))


models = [
("spaceship/models/Mesh043.obj", mat_Black, float3(20.0, 3.0, 3.0)),
("spaceship/models/Mesh029.obj", mat_Null, float3(3.88628, 3.77255, 3.43137)),
("spaceship/models/Mesh050.obj", mat_Backdrop),
("spaceship/models/Mesh042.obj", mat_RoughAluminium),
("spaceship/models/Mesh044.obj", mat_Leather),
("spaceship/models/Mesh038.obj", mat_RoughAluminium),
("spaceship/models/Mesh040.obj", mat_RoughAluminium),
("spaceship/models/Mesh072.obj", mat_RoughAluminium),
("spaceship/models/Mesh033.obj", mat_RoughSteel),
("spaceship/models/Mesh028.obj", mat_Leather),
("spaceship/models/Mesh055.obj", mat_RedLeather),
("spaceship/models/Mesh053.obj", mat_DarkPlastic),
("spaceship/models/Mesh036.obj", mat_RedLeather),
("spaceship/models/Mesh035.obj", mat_PinkLeather),
("spaceship/models/Mesh031.obj", mat_RedLeather),
("spaceship/models/Mesh030.obj", mat_PinkLeather),
("spaceship/models/Mesh027.obj", mat_RoughAluminium),
("spaceship/models/Mesh064.obj", mat_RoughSteel),
("spaceship/models/Mesh058.obj", mat_RoughSteel),
("spaceship/models/Mesh080.obj", mat_Black),
("spaceship/models/Mesh067.obj", mat_RoughAluminium),
("spaceship/models/Mesh060.obj", mat_RoughAluminium),
("spaceship/models/Mesh026.obj", mat_RoughSteel),
("spaceship/models/Mesh047.obj", mat_RoughSteel),
("spaceship/models/Mesh061.obj", mat_DarkPlastic),
("spaceship/models/Mesh063.obj", mat_DarkRubber),
("spaceship/models/Mesh065.obj", mat_RoughAluminium),
("spaceship/models/Mesh048.obj", mat_RoughAluminium),
("spaceship/models/Mesh066.obj", mat_RoughAluminium),
("spaceship/models/Mesh068.obj", mat_DarkRubber),
("spaceship/models/Mesh071.obj", mat_RoughSteel),
("spaceship/models/Mesh046.obj", mat_RoughAluminium),
("spaceship/models/Mesh076.obj", mat_RoughAluminium),
("spaceship/models/Mesh059.obj", mat_RoughAluminium),
("spaceship/models/Mesh057.obj", mat_RoughAluminium),
("spaceship/models/Mesh062.obj", mat_RoughAluminium),
("spaceship/models/Mesh074.obj", mat_RoughAluminium),
("spaceship/models/Mesh075.obj", mat_RoughAluminium),
("spaceship/models/Mesh078.obj", mat_RoughAluminium),
("spaceship/models/Mesh081.obj", mat_RoughAluminium),
("spaceship/models/Mesh034.obj", mat_RoughAluminium),
("spaceship/models/Mesh084.obj", mat_RoughAluminium),
("spaceship/models/Mesh085.obj", mat_RoughAluminium),
("spaceship/models/Mesh073.obj", mat_RoughAluminium),
("spaceship/models/Mesh077.obj", mat_RoughAluminium),
("spaceship/models/Mesh087.obj", mat_RoughAluminium),
("spaceship/models/Mesh052.obj", mat_RoughAluminium),
("spaceship/models/Mesh025.obj", mat_RoughAluminium),
("spaceship/models/Mesh024.obj", mat_RoughAluminium),
("spaceship/models/Mesh086.obj", mat_RoughAluminium),
("spaceship/models/Mesh023.obj", mat_RoughAluminium),
("spaceship/models/Mesh021.obj", mat_RoughAluminium),
("spaceship/models/Mesh039.obj", mat_RoughAluminium),
("spaceship/models/Mesh020.obj", mat_RoughAluminium),
("spaceship/models/Mesh032.obj", mat_RoughAluminium),
("spaceship/models/Mesh019.obj", mat_RoughSteel),
("spaceship/models/Mesh018.obj", mat_RoughAluminium),
("spaceship/models/Mesh070.obj", mat_RoughAluminium),
("spaceship/models/Mesh016.obj", mat_RoughAluminium),
("spaceship/models/Mesh015.obj", mat_RoughAluminium),
("spaceship/models/Mesh054.obj", mat_RoughAluminium),
("spaceship/models/Mesh013.obj", mat_RoughAluminium),
("spaceship/models/Mesh079.obj", mat_RoughAluminium),
("spaceship/models/Mesh041.obj", mat_RoughAluminium),
("spaceship/models/Mesh012.obj", mat_RoughAluminium),
("spaceship/models/Mesh011.obj", mat_RoughAluminium),
("spaceship/models/Mesh083.obj", mat_RoughAluminium),
("spaceship/models/Mesh010.obj", mat_RoughAluminium),
("spaceship/models/Mesh069.obj", mat_RoughAluminium),
("spaceship/models/Mesh009.obj", mat_RoughAluminium),
("spaceship/models/Mesh007.obj", mat_RoughAluminium),
("spaceship/models/Mesh017.obj", mat_RoughAluminium),
("spaceship/models/Mesh006.obj", mat_RoughSteel),
("spaceship/models/Mesh037.obj", mat_RoughAluminium),
("spaceship/models/Mesh008.obj", mat_RoughSteel),
("spaceship/models/Mesh045.obj", mat_RoughSteel),
("spaceship/models/Mesh005.obj", mat_RoughAluminium),
("spaceship/models/Mesh004.obj", mat_RoughAluminium),
("spaceship/models/Mesh049.obj", mat_RoughSteel),
("spaceship/models/Mesh082.obj", mat_RoughSteel),
("spaceship/models/Mesh014.obj", mat_RoughSteel),
("spaceship/models/Mesh003.obj", mat_RoughSteel),
("spaceship/models/Mesh002.obj", mat_RoughAluminium),
("spaceship/models/Mesh051.obj", mat_RoughSteel),
("spaceship/models/Mesh001.obj", mat_Glass),
("spaceship/models/Mesh056.obj", mat_RoughAluminium),
("spaceship/models/Mesh000.obj", mat_BrightPinkLeather),
("spaceship/models/Mesh022.obj", mat_RedLeather),
("spaceship/models/square.obj", mat_Null, float3(9.0, 9.0, 10.0), lcapi.transpose(float4x4(
    2.049729824066162, -2.478330429767084e-07, -2.478330429767084e-07, 0.0,
    -7.981942644619267e-07, 2.78191176761311e-08, -0.6364250183105469, 5.144720077514648,
    7.981942644619267e-07, 0.6364250183105469, 2.7819021752861772e-08, 1.6107900142669678,
    0.0, 0.0, 0.0, 1.0
))),
("spaceship/models/square.obj", mat_Null, float3(9.0, 9.0, 10.0), lcapi.transpose(float4x4(
    2.049729824066162, -2.478330429767084e-07, -2.478330429767084e-07, 0.0,
    -7.981942644619267e-07, 2.78191176761311e-08, -0.6364250183105469, 5.144720077514648,
    7.981942644619267e-07, 0.6364250183105469, 2.7819021752861772e-08, -1.6107900142669678,
    0.0, 0.0, 0.0, 1.0
))),
("spaceship/models/square.obj", mat_Backdrop, None, lcapi.transpose(float4x4(
      39.23594284057617, 0.0, 0.0, 0.0,
    0.0, -39.2359504699707, -3.430115839364589e-06, 39.20000076293945,
    0.0, 3.430115839364589e-06, -39.2359504699707, -30.20800018310547,
    0.0, 0.0, 0.0, 1.0
)))
]



const_env_light = float3(0.3)

camera_pos = float3(-0.5196635723114014, 0.8170070052146912, 3.824389696121216)
camera_dir = float3(0.13595250248908997, -0.05167683959007263, -0.9893665909767151)
camera_up = float3(0.0, 1.0, 0.0)
camera_fov = 35.98339777135762

resolution = 1920, 1080
rr_depth = 2
max_depth = 8

