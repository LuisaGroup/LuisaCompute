#!/usr/bin/env python3
"""
build_diverse_dataset.py -- Constructs high-quality, realistic multimodal benchmark dataset.

Builds a curated benchmark set of 16 diverse, high-quality, realistic images across 8 domains:
  1) Natural realism (wildlife/animals/nature photography)
  2) Ultra-wide landscape (panoramic 2:1 aspect ratio)
  3) Urban architecture / street night scene (tall 2:3 vertical aspect ratio)
  4) Macro close-up photography (depth-of-field separation)
  5) Classical artistic painting (impressionist oil painting / watercolor)
  6) Digital illustration / anime (high-fidelity detailed art)
  7) Technical schematic / blueprint (electronic circuit PCB or CAD blueprint)
  8) Statistical data chart / infographic (multi-metric charts/visualizations)

Features:
- Downloads real-world photography and fine art from verified CDN sources (Unsplash / Wikimedia).
- Procedurally generates vector-rasterized technical blueprints and multi-panel charts with matplotlib & PIL.
- Provides robust deterministic procedural fallbacks for all images when running offline.
- Exports rich bilingual (en, zh) metadata to diverse_dataset.json with:
  short_caption, detailed_caption, hard_negatives, and task_prompt_retrieval.
"""

import argparse
import io
import json
import math
import os
import sys
import urllib.request
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIVERSE_DATASET_SPEC = [
    # -------------------------------------------------------------------------
    # Domain 1: Natural Realism (4:3 aspect ratio, 800x600)
    # -------------------------------------------------------------------------
    {
        "id": "sample_01_wildlife_lion",
        "category": "natural_realism",
        "filename": "sample_01_wildlife_lion.png",
        "aspect_ratio": "4:3",
        "width": 800,
        "height": 600,
        "url": "https://images.unsplash.com/photo-1546182990-dffeafbe841d?w=800&h=600&fit=crop&q=85",
        "annotations": {
            "short_caption": {
                "en": "A majestic adult male lion resting in the African savannah",
                "zh": "一只在非洲大草原上休憩的威严雄�?
            },
            "detailed_caption": {
                "en": "Close-up portrait of a powerful male lion lying on dry golden grass, featuring a dense golden-brown mane, piercing amber eyes, and natural sunlight illuminating its fur.",
                "zh": "一头雄壮的成年雄狮趴在金黄色草地上的特写肖像，拥有浓密的金褐色鬃毛与锐利的琥珀色眼睛，自然阳光照亮了它的胡须与面部细节�?
            },
            "hard_negatives": [
                {
                    "en": "A female lioness hunting an antelope across the plains",
                    "zh": "一只在平原上追逐猎捕羚羊的雌性母�?
                },
                {
                    "en": "A close portrait of a Bengal tiger with vivid orange and black stripes",
                    "zh": "一只具有鲜明橙黑相间条纹的孟加拉虎正面肖像"
                }
            ],
            "task_prompt_retrieval": {
                "en": "Represent this wildlife photograph for animal species retrieval",
                "zh": "为野生动物物种识别检索表征该图像"
            }
        }
    },
    {
        "id": "sample_02_autumn_deer",
        "category": "natural_realism",
        "filename": "sample_02_autumn_deer.png",
        "aspect_ratio": "4:3",
        "width": 800,
        "height": 600,
        "url": "https://images.unsplash.com/photo-1484406566174-9da000fda645?w=800&h=600&fit=crop&q=85",
        "annotations": {
            "short_caption": {
                "en": "A red deer stag standing alert among misty autumn trees",
                "zh": "晨雾弥漫的秋日森林中驻足警觉的马�?
            },
            "detailed_caption": {
                "en": "A magnificent red deer stag with large branching antlers standing between golden-orange autumn trees in a morning foggy forest, surrounded by fallen leaves.",
                "zh": "一头头戴巨大分叉鹿角的高大雄性马鹿，伫立在金橙色秋叶与晨雾交织的静谧林间，脚下铺满红褐落叶�?
            },
            "hard_negatives": [
                {
                    "en": "A herd of wild elk grazing in a wide green summer alpine meadow",
                    "zh": "一群在夏季绿色高山开阔草甸上吃草的野生麋鹿群"
                },
                {
                    "en": "A moose wading through a shallow northern swamp lake",
                    "zh": "一只涉水穿过北方浅沼泽湖泊的驼�?
                }
            ],
            "task_prompt_retrieval": {
                "en": "Represent this wildlife image for natural habitat retrieval",
                "zh": "为自然栖息地分类检索表征该图像"
            }
        }
    },

    # -------------------------------------------------------------------------
    # Domain 2: Ultra-Wide Landscape (2:1 aspect ratio, 1200x600)
    # -------------------------------------------------------------------------
    {
        "id": "sample_03_panoramic_mountain_lake",
        "category": "wide_landscape",
        "filename": "sample_03_panoramic_mountain_lake.png",
        "aspect_ratio": "2:1",
        "width": 1200,
        "height": 600,
        "url": "https://images.unsplash.com/photo-1486870591958-9b9d0d1dda99?w=1200&h=600&fit=crop&q=85",
        "annotations": {
            "short_caption": {
                "en": "An ultra-wide panoramic landscape of snow-capped mountains and a pristine glacial lake",
                "zh": "白雪覆盖的高耸山脉与深蓝纯净冰川湖泊的超宽全景风�?
            },
            "detailed_caption": {
                "en": "A wide 2:1 panoramic view of rugged snow-covered mountain peaks mirroring across the mirror-like surface of a calm turquoise glacial alpine lake under a crisp clear sky.",
                "zh": "2:1极宽画幅的雄伟雪山群峰全景，陡峭的山体在如镜面般宁静清澈的绿松石色冰川湖泊中倒映，天空清朗蔚蓝�?
            },
            "hard_negatives": [
                {
                    "en": "A tropical turquoise lagoon with coral reefs and white sand beach",
                    "zh": "拥有珊瑚礁和洁白细沙滩的蓝绿色热带潟湖海�?
                },
                {
                    "en": "A misty mountain gorge with a cascading roaring waterfall",
                    "zh": "水汽弥漫的峡谷深处奔流而下的轰鸣瀑布景观"
                }
            ],
            "task_prompt_retrieval": {
                "en": "Represent this panoramic image for landscape scenery retrieval",
                "zh": "为全景自然风光库检索表征该图像"
            }
        }
    },
    {
        "id": "sample_04_panoramic_desert_dunes",
        "category": "wide_landscape",
        "filename": "sample_04_panoramic_desert_dunes.png",
        "aspect_ratio": "2:1",
        "width": 1200,
        "height": 600,
        "url": "https://images.unsplash.com/photo-1509316785289-025f5b846b35?w=1200&h=600&fit=crop&q=85",
        "annotations": {
            "short_caption": {
                "en": "Vast panoramic sand dunes in a golden desert with dramatic ridgelines and ripples",
                "zh": "夕阳映照下辽阔干旱荒漠中延绵起伏的金色风蚀沙丘与沙�?
            },
            "detailed_caption": {
                "en": "Sweeping panoramic desert view showing sharp wind-carved ridges of golden and reddish sand dunes casting deep long shadows under an expansive glowing twilight sky.",
                "zh": "极具视觉冲击力的超宽沙漠远景，风力雕琢出的刀刃般沙脊向远方延伸，夕阳在金红色沙海上映照出狭长深邃的阴影�?
            },
            "hard_negatives": [
                {
                    "en": "A vast desolate snow-covered arctic tundra with wind-blown snow ridges under an overcast sky",
                    "zh": "阴云笼罩下一片白茫茫的荒凉北极苔原与风蚀雪脊"
                },
                {
                    "en": "A caravan of camels walking in single file across a dry salt flat",
                    "zh": "一队单峰骆驼排成单列穿行在白茫茫的干涸盐沼地上"
                }
            ],
            "task_prompt_retrieval": {
                "en": "Represent this panoramic landscape for arid terrain retrieval",
                "zh": "为干旱地貌遥感与地理场景检索表征该图像"
            }
        }
    },

    # -------------------------------------------------------------------------
    # Domain 3: Urban Architecture / Street Night (2:3 aspect ratio, 720x1080)
    # -------------------------------------------------------------------------
    {
        "id": "sample_05_rainy_city_night",
        "category": "urban_architecture",
        "filename": "sample_05_rainy_city_night.png",
        "aspect_ratio": "2:3",
        "width": 720,
        "height": 1080,
        "url": "https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=720&h=1080&fit=crop&q=85",
        "annotations": {
            "short_caption": {
                "en": "A vertical night view of a rainy city street illuminated by vibrant neon signs",
                "zh": "霓虹灯绚丽闪烁的雨夜繁华都市街头纵向夜景"
            },
            "detailed_caption": {
                "en": "Vertical 2:3 street photography of a bustling urban intersection at night during rainfall, with colorful neon advertisements and headlights reflecting off wet glossy asphalt.",
                "zh": "2:3竖构图的雨夜都市街景，两旁密布高耸建筑与发光霓虹广告牌，来往车流前灯与斑斓招牌倒映在湿漉漉的反光沥青路面上�?
            },
            "hard_negatives": [
                {
                    "en": "A sunny daytime avenue with pedestrians walking past historic stone buildings",
                    "zh": "阳光明媚的白天林荫大道，行人穿梭在古典石质欧洲建筑群�?
                },
                {
                    "en": "An empty subway station platform illuminated by cold fluorescent tube lights",
                    "zh": "冷白色荧光灯管照耀下空无一人的地下地铁站台"
                }
            ],
            "task_prompt_retrieval": {
                "en": "Represent this urban photo for city nightscape retrieval",
                "zh": "为城市夜景与街区视觉库检索表征该图像"
            }
        }
    },
    {
        "id": "sample_06_skyscraper_facade",
        "category": "urban_architecture",
        "filename": "sample_06_skyscraper_facade.png",
        "aspect_ratio": "2:3",
        "width": 720,
        "height": 1080,
        "url": "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=720&h=1080&fit=crop&q=85",
        "annotations": {
            "short_caption": {
                "en": "A low-angle vertical shot looking up at towering modern glass skyscrapers",
                "zh": "自街头仰拍高耸入云的现代玻璃幕墙摩天大楼纵向构图"
            },
            "detailed_caption": {
                "en": "Vertical low-angle architectural perspective looking up into the sky between reflective glass curtain wall skyscrapers, showing converging geometric grid lines and cloud reflections.",
                "zh": "极具透视感的纵向低角度仰拍，两侧巨大的现代化双曲面玻璃幕墙摩天大厦向蓝天顶部收拢，金属窗框网格清晰，映衬着漂浮白云�?
            },
            "hard_negatives": [
                {
                    "en": "An ancient Gothic stone cathedral with pointed arches and stained glass",
                    "zh": "拥有尖拱门与彩色琉璃花窗的古老哥特式石头大教�?
                },
                {
                    "en": "A quiet residential street of two-story brick suburban houses with gardens",
                    "zh": "带有前院绿植花园的红砖两层郊区联排住宅街�?
                }
            ],
            "task_prompt_retrieval": {
                "en": "Represent this architectural perspective for building structure retrieval",
                "zh": "为现代建筑透视与工程立面检索表征该图像"
            }
        }
    },

    # -------------------------------------------------------------------------
    # Domain 4: Macro Close-up Photography (1:1 aspect ratio, 600x600)
    # -------------------------------------------------------------------------
    {
        "id": "sample_07_macro_dewdrop",
        "category": "macro_close_up",
        "filename": "sample_07_macro_dewdrop.png",
        "aspect_ratio": "1:1",
        "width": 600,
        "height": 600,
        "url": "https://images.unsplash.com/photo-1533038590840-1cde6e668a91?w=600&h=600&fit=crop&q=85",
        "annotations": {
            "short_caption": {
                "en": "A crystal clear morning dewdrop clinging to a fresh green leaf",
                "zh": "晶莹剔透附着在嫩绿叶片上的晨露微距特�?
            },
            "detailed_caption": {
                "en": "Square extreme macro photography of a spherical transparent water droplet resting delicately on the serrated edge of a vibrant green plant leaf, with creamy blurred bokeh in the background.",
                "zh": "1:1正方形超微距摄影，一颗浑圆通透的露珠静止在翠绿叶片尖端，微小植物脉络纤维清晰可辨，背景呈现柔和纯净的浅绿色虚化光斑�?
            },
            "hard_negatives": [
                {
                    "en": "Heavy rain pouring on a window glass pane during a severe thunderstorm",
                    "zh": "狂风暴雨天大量雨水猛烈冲刷拍打在透明窗户玻璃�?
                },
                {
                    "en": "Ice crystals and frost feathers growing across frozen lake ice",
                    "zh": "在冰冻湖面结晶生长的羽毛状冰霜结晶花�?
                }
            ],
            "task_prompt_retrieval": {
                "en": "Represent this macro photo for botanical detail retrieval",
                "zh": "为植物微观结构与自然微距库检索表征该图像"
            }
        }
    },
    {
        "id": "sample_08_macro_honeybee",
        "category": "macro_close_up",
        "filename": "sample_08_macro_honeybee.png",
        "aspect_ratio": "1:1",
        "width": 600,
        "height": 600,
        "url": "https://images.unsplash.com/photo-1558642452-9d2a7deb7f62?w=600&h=600&fit=crop&q=85",
        "annotations": {
            "short_caption": {
                "en": "A macro close-up of a honeybee gathering golden pollen on a blooming flower",
                "zh": "在盛开花盘上采集金黄花粉的蜜蜂超微距特�?
            },
            "detailed_caption": {
                "en": "Detailed square macro shot of a striped honeybee dusted with yellow pollen grains clinging to the textured center disc of a vivid yellow sunflower, with translucent wing veins visible.",
                "zh": "高清晰度微距摄影，一只带有黑褐绒毛条纹的工蜂停留在向日葵中央，足部花粉囊附满金黄色花粉粒，翅膀薄膜上的微细脉络一览无余�?
            },
            "hard_negatives": [
                {
                    "en": "A bright orange monarch butterfly perched on purple lavender blossoms",
                    "zh": "一只停留在紫色薰衣草花丛上的鲜艳橙黑斑纹黑脉金斑蝶"
                },
                {
                    "en": "A ladybug beetle with black spots crawling along a smooth red rose petal",
                    "zh": "一只背负黑斑圆点的红甲七星瓢虫沿着红玫瑰花瓣爬�?
                }
            ],
            "task_prompt_retrieval": {
                "en": "Represent this macro insect photo for entomological retrieval",
                "zh": "为昆虫生态学微距观察检索表征该图像"
            }
        }
    },

    # -------------------------------------------------------------------------
    # Domain 5: Classical Artistic Painting (4:3 aspect ratio, 800x600)
    # -------------------------------------------------------------------------
    {
        "id": "sample_09_impressionist_waterlilies",
        "category": "classical_painting",
        "filename": "sample_09_impressionist_waterlilies.png",
        "aspect_ratio": "4:3",
        "width": 800,
        "height": 600,
        "url": "https://upload.wikimedia.org/wikipedia/commons/e/e7/Monet_Water_lilies_1907.jpg",
        "annotations": {
            "short_caption": {
                "en": "An impressionist oil painting of water lilies floating on a sunlit pond",
                "zh": "阳光下水波荡漾的睡莲池塘印象派油�?
            },
            "detailed_caption": {
                "en": "Classical impressionist oil artwork depicting blooming white and pink water lilies on a tranquil pond surface, rendered with visible textured impasto brushstrokes and dappled pastel reflections.",
                "zh": "经典印象派油画作品，水塘表面盛开着粉白相间的睡莲花朵，运用了富有立体肌理的厚涂笔触与柔和色彩，展现出浮光掠影的光影变幻�?
            },
            "hard_negatives": [
                {
                    "en": "A classical Renaissance oil portrait of an Elizabethan nobleman in dark velvet",
                    "zh": "身着深色天鹅绒华服的古典文艺复兴贵族油画肖像"
                },
                {
                    "en": "A dramatic oil painting of stormy ocean waves crashing violently into sea cliffs",
                    "zh": "狂风暴雨中巨浪猛烈撞击黑色海岸悬崖的写实油画"
                }
            ],
            "task_prompt_retrieval": {
                "en": "Represent this fine art painting for art history retrieval",
                "zh": "为艺术史古典绘画典藏检索表征该图像"
            }
        }
    },
    {
        "id": "sample_10_starry_night",
        "category": "classical_painting",
        "filename": "sample_10_starry_night.png",
        "aspect_ratio": "4:3",
        "width": 800,
        "height": 600,
        "url": "https://upload.wikimedia.org/wikipedia/commons/0/01/Vincent_van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
        "annotations": {
            "short_caption": {
                "en": "A post-impressionist oil painting of a swirling starry night sky over a village",
                "zh": "夜空中星云剧烈旋转的后印象派经典星空油画"
            },
            "detailed_caption": {
                "en": "Post-impressionist masterpiece painting featuring intense swirling spirals of yellow and cobalt blue across a dark night sky, with glowing stars and a dark cypress tree silhouette.",
                "zh": "后印象派代表性油画杰作，深蓝夜空中流转着黄蓝相间的剧烈旋转旋涡星云，耀眼的星光、弯月与前景高耸的深色柏树轮廓交织�?
            },
            "hard_negatives": [
                {
                    "en": "An astrophotography long-exposure photo of the Milky Way over a desert observatory",
                    "zh": "天文摄影长曝光拍摄的沙漠天文台上空绚烂银河拱桥实�?
                },
                {
                    "en": "A clean minimalist digital vector icon of the moon and stars",
                    "zh": "扁平极简风格的月亮与星星数字化矢量小图标"
                }
            ],
            "task_prompt_retrieval": {
                "en": "Represent this expressive artwork for museum collection retrieval",
                "zh": "为博物馆名画藏品艺术风格检索表征该图像"
            }
        }
    },

    # -------------------------------------------------------------------------
    # Domain 6: Digital Illustration / Concept Art (1:1 aspect ratio, 896x896)
    # -------------------------------------------------------------------------
    {
        "id": "sample_11_cyberpunk_anime",
        "category": "digital_illustration",
        "filename": "sample_11_cyberpunk_anime.png",
        "aspect_ratio": "1:1",
        "width": 896,
        "height": 896,
        "url": None,
        "annotations": {
            "short_caption": {
                "en": "A cyberpunk anime character standing in front of a futuristic holographic city window",
                "zh": "站在未来全息都市观景窗前的赛博朋克动漫人物插�?
            },
            "detailed_caption": {
                "en": "High-fidelity digital 2D illustration in modern anime aesthetics depicting a cyber-enhanced character standing beside a towering bay window, overlooking neon skyscrapers and floating vehicles in rain.",
                "zh": "高保真现代二次元数码概念插画，身着机械义体风衣的科幻人物伫立在落地全景窗前，俯瞰窗外霓虹悬浮车流穿梭的雨夜赛博都市�?
            },
            "hard_negatives": [
                {
                    "en": "A retro pixel art 8-bit platformer game screenshot in a dungeon",
                    "zh": "地牢迷宫题材的复�?位像素风格横版过关游戏画�?
                },
                {
                    "en": "A traditional Japanese Ukiyo-e woodblock print of Mount Fuji",
                    "zh": "展现富士山景色的日本传统浮世绘木版水印画"
                }
            ],
            "task_prompt_retrieval": {
                "en": "Represent this concept illustration for digital media retrieval",
                "zh": "为数字创意概念设计图库检索表征该图像"
            }
        }
    },
    {
        "id": "sample_12_fantasy_castle",
        "category": "digital_illustration",
        "filename": "sample_12_fantasy_castle.png",
        "aspect_ratio": "1:1",
        "width": 896,
        "height": 896,
        "url": "https://upload.wikimedia.org/wikipedia/commons/a/ae/Castle_Neuschwanstein.jpg",
        "annotations": {
            "short_caption": {
                "en": "A fairytale alpine castle with majestic soaring white towers and turrets",
                "zh": "坐落在高山峰峦间耸立着洁白尖塔与角楼的童话城堡"
            },
            "detailed_caption": {
                "en": "Picturesque fairytale fortress with dramatic soaring white towers, turrets and spires surrounded by autumn alpine forest and rugged mountain peaks.",
                "zh": "如童话般巍峨梦幻的高山城堡，拔地而起的高耸白色尖塔与八角塔楼依山而建，掩映在色彩斑斓的秋日森林与崇山峻岭之中�?
            },
            "hard_negatives": [
                {
                    "en": "A historical black and white aerial photograph of a medieval fortress in Germany",
                    "zh": "俯拍德国中世纪石砌城堡历史黑白航拍真实照�?
                },
                {
                    "en": "A children crayon drawing of a smiling house with a chimney",
                    "zh": "带有烟囱和笑脸太阳的儿童蜡笔涂鸦小房�?
                }
            ],
            "task_prompt_retrieval": {
                "en": "Represent this castle image for architectural landmark retrieval",
                "zh": "为著名建筑地标库检索表征该图像"
            }
        }
    },

    # -------------------------------------------------------------------------
    # Domain 7: Technical Schematic / Blueprint (4:3 aspect ratio, 1024x768)
    # -------------------------------------------------------------------------
    {
        "id": "sample_13_pcb_blueprint",
        "category": "technical_schematic",
        "filename": "sample_13_pcb_blueprint.png",
        "aspect_ratio": "4:3",
        "width": 1024,
        "height": 768,
        "url": None,  # Generated procedurally
        "annotations": {
            "short_caption": {
                "en": "A detailed electronic circuit PCB blueprint schematic diagram",
                "zh": "一张精细的电子电路印刷板PCB原理图蓝�?
            },
            "detailed_caption": {
                "en": "High-density electronic circuit schematic diagram with blue blueprint background, copper tracing routes, capacitors, microchips and solder pads clearly labeled.",
                "zh": "深蓝色背景上的高密度集成电路原理蓝图，清晰展示铜导线布线、贴片电容、微控制器引脚以及测试焊盘的工程结构�?
            },
            "hard_negatives": [
                {
                    "en": "A blue architectural floor plan of a modern two-story residential house",
                    "zh": "一张蓝色背景的现代两层住宅建筑户型平面规划设计�?
                },
                {
                    "en": "A mechanical assembly CAD drawing showing engine gears and transmission belts",
                    "zh": "展示发动机精密齿轮与传动皮带装配关系的机械CAD零件�?
                }
            ],
            "task_prompt_retrieval": {
                "en": "Represent this image for technical document retrieval",
                "zh": "为技术文档检索表征该图像"
            }
        }
    },
    {
        "id": "sample_14_cad_blueprint",
        "category": "technical_schematic",
        "filename": "sample_14_cad_blueprint.png",
        "aspect_ratio": "4:3",
        "width": 1024,
        "height": 768,
        "url": None,  # Generated procedurally
        "annotations": {
            "short_caption": {
                "en": "A mechanical engineering CAD assembly blueprint of a precision gearbox",
                "zh": "精密减速齿轮箱机械工程CAD装配蓝图"
            },
            "detailed_caption": {
                "en": "Engineering mechanical blueprint on dark blueprint cyan grid featuring technical orthographic cross-section views, dimensional tolerance annotations, spur gears, roller bearings, and drive shafts.",
                "zh": "深青色网格工程蓝图背景上的精密齿轮箱装配图，展现三视图正交剖面结构，标注配合公差尺寸、正齿轮啮合齿形、深沟球轴承与动力传动轴�?
            },
            "hard_negatives": [
                {
                    "en": "An electrical single-line wiring diagram for a high-voltage power substation",
                    "zh": "高压变电站配电系统的电气单线一次接线原理图"
                },
                {
                    "en": "A civil engineering highway interchange bridge elevation profile blueprint",
                    "zh": "公路立交枢纽特大桥梁下部结构立面标高施工蓝图"
                }
            ],
            "task_prompt_retrieval": {
                "en": "Represent this blueprint for mechanical patent retrieval",
                "zh": "为机械工程专利图纸检索表征该图像"
            }
        }
    },

    # -------------------------------------------------------------------------
    # Domain 8: Statistical Data Chart / Infographic (3:2 aspect ratio, 960x640)
    # -------------------------------------------------------------------------
    {
        "id": "sample_15_business_dashboard",
        "category": "statistical_chart",
        "filename": "sample_15_business_dashboard.png",
        "aspect_ratio": "3:2",
        "width": 960,
        "height": 640,
        "url": None,  # Generated procedurally
        "annotations": {
            "short_caption": {
                "en": "A comprehensive business analytics dashboard with bar charts and KPI metrics",
                "zh": "包含柱状图与多项关键KPI指标的企业商业数据分析看�?
            },
            "detailed_caption": {
                "en": "Structured executive business analytics dashboard featuring multiple data visualizations: monthly revenue bar charts, conversion rate line plots, regional sales breakdown pie charts, and summary metric cards.",
                "zh": "结构清晰的企业经营多指标数据分析看板，集成月度营业收入对比柱状图、转化率趋势折线图、各区域市场销售占比环形图以及核心KPI指标卡片�?
            },
            "hard_negatives": [
                {
                    "en": "A periodic table of chemical elements with atomic weights and group numbers",
                    "zh": "标注文明原子量与主族序号的化学元素周期表挂图"
                },
                {
                    "en": "A world transit network map showing international flight routes",
                    "zh": "标注全球各大主要枢纽机场洲际航线与航班班次的世界交通网络图"
                }
            ],
            "task_prompt_retrieval": {
                "en": "Represent this infographic for business report retrieval",
                "zh": "为商业决策报表与经营看板检索表征该图像"
            }
        }
    },
    {
        "id": "sample_16_scientific_infographic",
        "category": "statistical_chart",
        "filename": "sample_16_scientific_infographic.png",
        "aspect_ratio": "3:2",
        "width": 960,
        "height": 640,
        "url": None,  # Generated procedurally
        "annotations": {
            "short_caption": {
                "en": "A scientific research infographic displaying multi-panel climate data and heatmaps",
                "zh": "展示多组气候监测数据与温度热力图的科学科研信息�?
            },
            "detailed_caption": {
                "en": "Academic scientific visualization figure containing four subplots: 2D temperature anomaly contour heatmap, CO2 concentration scatter plot with regression curves, polar wind velocity vector field, and statistical box plots with error bars.",
                "zh": "四联学术科研数据图版，分别呈现全球地表温度异常二维等高线热力图、大气二氧化碳浓度随时间演变散点拟合曲线、极地风速向量场以及带误差棒的箱线统计图�?
            },
            "hard_negatives": [
                {
                    "en": "An infographic poster showing human digestive anatomy and organ functions",
                    "zh": "解析人体消化系统生理结构与器官机能的人体解剖科普信息海报"
                },
                {
                    "en": "A subway system route map with multi-colored interconnected metro lines",
                    "zh": "由红蓝绿多色换乘路线相互交织连接的城市现代地铁轨道交通线路图"
                }
            ],
            "task_prompt_retrieval": {
                "en": "Represent this scientific figure for scholarly literature retrieval",
                "zh": "为学术科技论文图版检索表征该图像"
            }
        }
    }
]


# =============================================================================
# Procedural Generators (Deterministic & High-Fidelity)
# =============================================================================

def generate_pcb_blueprint(width=1024, height=768) -> Image.Image:
    """Generates an authentic electronic circuit PCB schematic blueprint."""
    img = Image.new("RGB", (width, height), color=(10, 28, 56))  # Blueprint dark navy
    draw = ImageDraw.Draw(img)

    # 1. Blueprint background grid
    grid_spacing = 24
    grid_color = (18, 48, 88)
    for x in range(0, width, grid_spacing):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for y in range(0, height, grid_spacing):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)

    # Border frame
    margin = 32
    draw.rectangle([margin, margin, width - margin, height - margin], outline=(64, 160, 230), width=3)
    draw.rectangle([margin + 6, margin + 6, width - margin - 6, height - margin - 6], outline=(40, 100, 160), width=1)

    # Title block in bottom right
    tb_w, tb_h = 320, 110
    tb_x0, tb_y0 = width - margin - tb_w, height - margin - tb_h
    draw.rectangle([tb_x0, tb_y0, width - margin, height - margin], fill=(8, 22, 45), outline=(64, 160, 230), width=2)
    draw.line([(tb_x0, tb_y0 + 35), (width - margin, tb_y0 + 35)], fill=(64, 160, 230), width=1)
    draw.line([(tb_x0, tb_y0 + 70), (width - margin, tb_y0 + 70)], fill=(64, 160, 230), width=1)
    draw.text((tb_x0 + 12, tb_y0 + 10), "CIRCUIT SCHEMATIC REV 4.2", fill=(120, 210, 255))
    draw.text((tb_x0 + 12, tb_y0 + 45), "SYSTEM: 32-BIT MCU COPROCESSOR", fill=(200, 230, 255))
    draw.text((tb_x0 + 12, tb_y0 + 80), "SHEET: 1 OF 1  |  SCALE: 1:1  |  STATUS: PASSED", fill=(100, 180, 230))

    # 2. Main IC packages (Microcontroller U1 & Memory U2)
    def draw_ic(cx, cy, w, h, label, num_pins_per_side=8):
        x0, y0 = cx - w // 2, cy - h // 2
        x1, y1 = cx + w // 2, cy + h // 2
        # IC Body
        draw.rectangle([x0, y0, x1, y1], fill=(16, 42, 80), outline=(100, 200, 255), width=2)
        # Orientation notch
        draw.arc([x0 + 10, y0 - 8, x0 + 26, y0 + 8], 0, 180, fill=(100, 200, 255), width=2)
        draw.text((cx - 30, cy - 8), label, fill=(255, 255, 255))
        # Pins
        pin_len = 16
        pin_spacing = h / (num_pins_per_side + 1)
        for i in range(1, num_pins_per_side + 1):
            py = int(y0 + i * pin_spacing)
            # Left pins
            draw.line([(x0 - pin_len, py), (x0, py)], fill=(200, 220, 255), width=2)
            draw.ellipse([x0 - pin_len - 3, py - 3, x0 - pin_len + 3, py + 3], fill=(240, 200, 80))
            # Right pins
            draw.line([(x1, py), (x1 + pin_len, py)], fill=(200, 220, 255), width=2)
            draw.ellipse([x1 + pin_len - 3, py - 3, x1 + pin_len + 3, py + 3], fill=(240, 200, 80))

    draw_ic(380, 320, 160, 200, "U1: STM32F4", num_pins_per_side=10)
    draw_ic(720, 320, 130, 160, "U2: W25Q128", num_pins_per_side=8)

    # 3. Interconnecting traces with 45-degree bends & vias
    traces = [
        # Bus between U1 and U2
        ([(460 + 16, 260), (520, 260), (580, 260), (720 - 65 - 16, 260)], (80, 220, 240)),
        ([(460 + 16, 280), (530, 280), (570, 290), (720 - 65 - 16, 290)], (80, 220, 240)),
        ([(460 + 16, 300), (540, 300), (580, 320), (720 - 65 - 16, 320)], (80, 220, 240)),
        ([(460 + 16, 320), (550, 320), (590, 350), (720 - 65 - 16, 350)], (80, 220, 240)),
        # Traces to connector on left
        ([(300 - 16, 250), (200, 250), (140, 250)], (240, 180, 60)),
        ([(300 - 16, 275), (210, 275), (170, 315), (140, 315)], (240, 180, 60)),
        ([(300 - 16, 300), (220, 300), (180, 360), (140, 360)], (240, 180, 60)),
        ([(300 - 16, 350), (220, 350), (190, 410), (140, 410)], (240, 180, 60)),
        ([(300 - 16, 375), (230, 375), (200, 455), (140, 455)], (240, 180, 60)),
        # Power rail traces on top
        ([(380, 220), (380, 140), (720, 140), (720, 240)], (255, 100, 100)),
        ([(350, 220), (350, 120), (840, 120), (840, 450), (785 + 16, 450)], (100, 255, 120)),
    ]
    for pts, color in traces:
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=color, width=2)
            # Add vias at corners
            draw.ellipse([pts[i][0] - 3, pts[i][1] - 3, pts[i][0] + 3, pts[i][1] + 3], fill=(240, 200, 80), outline=(255, 255, 255))

    # 4. Surface mount passive components (Capacitors C1-C8, Resistors R1-R6)
    def draw_smd_resistor(x, y, label):
        draw.rectangle([x, y, x + 24, y + 12], fill=(40, 40, 40), outline=(180, 180, 180), width=1)
        draw.rectangle([x, y, x + 5, y + 12], fill=(220, 220, 220))
        draw.rectangle([x + 19, y, x + 24, y + 12], fill=(220, 220, 220))
        draw.text((x - 2, y - 14), label, fill=(200, 230, 255))

    def draw_smd_cap(x, y, label):
        draw.rectangle([x, y, x + 20, y + 12], fill=(180, 130, 80), outline=(200, 160, 100), width=1)
        draw.rectangle([x, y, x + 4, y + 12], fill=(220, 220, 220))
        draw.rectangle([x + 16, y, x + 20, y + 12], fill=(220, 220, 220))
        draw.text((x - 2, y - 14), label, fill=(200, 230, 255))

    draw_smd_cap(480, 134, "C1 100nF")
    draw_smd_cap(580, 134, "C2 10uF")
    draw_smd_cap(660, 134, "C3 100nF")
    draw_smd_resistor(170, 244, "R1 10k")
    draw_smd_resistor(180, 309, "R2 4.7k")
    draw_smd_resistor(190, 354, "R3 4.7k")

    # Connector Header J1
    draw.rectangle([100, 220, 136, 480], fill=(12, 34, 68), outline=(120, 200, 255), width=2)
    draw.text((105, 195), "J1: DEBUG", fill=(200, 230, 255))
    for y in range(240, 470, 24):
        draw.rectangle([112, y, 124, y + 12], fill=(240, 200, 80), outline=(255, 255, 255))

    return img


def generate_cad_blueprint(width=1024, height=768) -> Image.Image:
    """Generates an engineering mechanical assembly blueprint of a gearbox."""
    img = Image.new("RGB", (width, height), color=(8, 26, 48))
    draw = ImageDraw.Draw(img)

    # Grid
    grid_spacing = 32
    for x in range(0, width, grid_spacing):
        draw.line([(x, 0), (x, height)], fill=(14, 42, 75), width=1)
    for y in range(0, height, grid_spacing):
        draw.line([(0, y), (width, y)], fill=(14, 42, 75), width=1)

    # Frame & Title Block
    m = 28
    draw.rectangle([m, m, width - m, height - m], outline=(80, 180, 240), width=2)
    draw.rectangle([width - m - 340, height - m - 100, width - m, height - m], outline=(80, 180, 240), width=2, fill=(6, 20, 38))
    draw.text((width - m - 325, height - m - 85), "ISO 128 MECHANICAL ASSEMBLY", fill=(100, 210, 255))
    draw.text((width - m - 325, height - m - 60), "PLANETARY GEARBOX SPEED REDUCER", fill=(240, 240, 255))
    draw.text((width - m - 325, height - m - 35), "PART NO: GB-2026-X8 | TOLERANCE: +/-0.01", fill=(120, 180, 220))

    # Centerlines
    cx, cy = 460, 360
    draw.line([(cx - 280, cy), (cx + 280, cy)], fill=(120, 160, 200), width=1)
    draw.line([(cx, cy - 260), (cx, cy + 260)], fill=(120, 160, 200), width=1)

    # Gear 1 (Outer Ring Gear)
    draw.ellipse([cx - 220, cy - 220, cx + 220, cy + 220], outline=(100, 220, 255), width=3)
    draw.ellipse([cx - 200, cy - 200, cx + 200, cy + 200], outline=(70, 160, 210), width=1)  # Pitch circle

    # Gear teeth profile simulation
    num_teeth = 36
    for i in range(num_teeth):
        angle = 2 * math.pi * i / num_teeth
        x_in = cx + 205 * math.cos(angle)
        y_in = cy + 205 * math.sin(angle)
        x_out = cx + 220 * math.cos(angle)
        y_out = cy + 220 * math.sin(angle)
        draw.line([(x_in, y_in), (x_out, y_out)], fill=(140, 220, 255), width=2)

    # Sun Gear (Center)
    draw.ellipse([cx - 70, cy - 70, cx + 70, cy + 70], outline=(100, 220, 255), width=2)
    draw.ellipse([cx - 40, cy - 40, cx + 40, cy + 40], outline=(80, 180, 230), width=2)  # Central bore
    draw.rectangle([cx - 8, cy - 48, cx + 8, cy - 35], fill=(8, 26, 48), outline=(100, 220, 255), width=1)  # Keyway

    # 3 Planet Gears
    for k in range(3):
        pa = 2 * math.pi * k / 3 + math.pi / 6
        px = cx + 135 * math.cos(pa)
        py = cy + 135 * math.sin(pa)
        draw.ellipse([px - 65, py - 65, px + 65, py + 65], outline=(100, 220, 255), width=2)
        draw.ellipse([px - 25, py - 25, px + 25, py + 25], outline=(80, 180, 230), width=1)
        # Bearing rollers
        for b in range(8):
            ba = 2 * math.pi * b / 8
            bx = px + 18 * math.cos(ba)
            by = py + 18 * math.sin(ba)
            draw.ellipse([bx - 4, by - 4, bx + 4, by + 4], fill=(200, 230, 255))

    # Cross-hatching for section view in upper-left
    hx0, hy0, hx1, hy1 = 80, 80, 280, 220
    draw.rectangle([hx0, hy0, hx1, hy1], outline=(80, 180, 240), width=2, fill=(10, 32, 60))
    draw.text((hx0 + 10, hy0 + 10), "SECTION A-A: BEARING HOUSING", fill=(140, 210, 255))
    for s in range(hx0 - 100, hx1 + 100, 16):
        draw.line([(s, hy0), (s + (hy1 - hy0), hy1)], fill=(20, 60, 110), width=1)

    # Technical Dimension Leaders
    def draw_dimension(x1, y1, x2, y2, text, offset=25):
        # Leader line
        draw.line([(x1, y1 - offset), (x2, y2 - offset)], fill=(240, 200, 80), width=1)
        draw.line([(x1, y1), (x1, y1 - offset - 5)], fill=(240, 200, 80), width=1)
        draw.line([(x2, y2), (x2, y2 - offset - 5)], fill=(240, 200, 80), width=1)
        draw.text(((x1 + x2) // 2 - 25, y1 - offset - 15), text, fill=(255, 230, 100))

    draw_dimension(cx - 220, cy - 220, cx + 220, cy - 220, "O 440.00 H7", offset=30)
    draw_dimension(cx - 70, cy - 70, cx + 70, cy - 70, "O 140.00 +/-0.02", offset=20)
    draw_dimension(80, 220, 280, 220, "200.00 mm", offset=20)

    return img


def generate_business_dashboard(width=960, height=640) -> Image.Image:
    """Generates an executive statistical business dashboard with charts."""
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="#0f172a")

    # Layout with GridSpec: Title top, 4 KPI cards row, then 3 subplots below
    gs = fig.add_gridspec(3, 3, height_ratios=[0.5, 1.2, 1.3], hspace=0.38, wspace=0.32,
                           left=0.06, right=0.95, top=0.92, bottom=0.08)

    # Global text colors
    plt.rcParams["text.color"] = "#f1f5f9"
    plt.rcParams["axes.labelcolor"] = "#94a3b8"
    plt.rcParams["xtick.color"] = "#94a3b8"
    plt.rcParams["ytick.color"] = "#94a3b8"

    # Header title
    fig.text(0.06, 0.95, "Enterprise Global Performance & Revenue Analytics Dashboard",
             fontsize=14, fontweight="bold", color="#f8fafc")
    fig.text(0.06, 0.925, "Quarterly Q3 Consolidated Business Metrics & Growth Projections",
             fontsize=9, color="#64748b")

    # 1. Monthly Revenue Bar Chart
    ax1 = fig.add_subplot(gs[1, 0:2], facecolor="#1e293b")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    revenue = [2.4, 2.7, 3.1, 3.0, 3.6, 4.2, 4.5, 4.8, 5.2, 5.8, 6.1, 6.7]
    target = [2.5, 2.8, 3.0, 3.2, 3.5, 3.8, 4.0, 4.4, 4.8, 5.2, 5.6, 6.0]
    bars = ax1.bar(months, revenue, color="#38bdf8", width=0.6, label="Actual Revenue ($M)", alpha=0.9)
    ax1.plot(months, target, color="#f59e0b", marker="o", linewidth=2, label="Budget Target ($M)")
    ax1.set_title("Monthly Recurring Revenue ($M)", fontsize=10, fontweight="bold", pad=8)
    ax1.legend(loc="upper left", fontsize=7, facecolor="#0f172a", edgecolor="#334155")
    ax1.grid(True, linestyle="--", alpha=0.2, color="#94a3b8")

    # 2. Regional Sales Donut Chart
    ax2 = fig.add_subplot(gs[1, 2], facecolor="#1e293b")
    regions = ["North America", "EMEA", "Asia Pacific", "Latin America"]
    shares = [42, 28, 22, 8]
    colors = ["#38bdf8", "#818cf8", "#34d399", "#fbbf24"]
    wedges, texts, autotexts = ax2.pie(
        shares, labels=regions, autopct="%1.0f%%", startangle=140,
        colors=colors, textprops={"fontsize": 7},
        wedgeprops=dict(width=0.45, edgecolor="#0f172a", linewidth=2)
    )
    for at in autotexts:
        at.set_color("#ffffff")
        at.set_fontweight("bold")
    ax2.set_title("Market Share by Region", fontsize=10, fontweight="bold", pad=8)

    # 3. Conversion Rate & Retention Trend (Multi-series)
    ax3 = fig.add_subplot(gs[2, 0:2], facecolor="#1e293b")
    weeks = [f"W{i}" for i in range(1, 13)]
    funnel_a = [3.2, 3.4, 3.3, 3.6, 3.9, 4.1, 4.0, 4.3, 4.6, 4.8, 5.1, 5.4]
    funnel_b = [2.1, 2.2, 2.4, 2.5, 2.7, 2.6, 2.9, 3.1, 3.0, 3.3, 3.5, 3.8]
    ax3.plot(weeks, funnel_a, color="#34d399", marker="s", markersize=4, linewidth=2, label="Organic Funnel (%)")
    ax3.plot(weeks, funnel_b, color="#a78bfa", marker="^", markersize=4, linewidth=2, label="Paid Funnel (%)")
    ax3.fill_between(weeks, funnel_b, funnel_a, color="#38bdf8", alpha=0.12)
    ax3.set_title("Weekly Conversion Rate Trajectory (%)", fontsize=10, fontweight="bold", pad=8)
    ax3.legend(loc="upper left", fontsize=7, facecolor="#0f172a", edgecolor="#334155")
    ax3.grid(True, linestyle="--", alpha=0.2, color="#94a3b8")

    # 4. Acquisition Channel Breakdown (Horizontal Bar)
    ax4 = fig.add_subplot(gs[2, 2], facecolor="#1e293b")
    channels = ["Direct", "Search SEO", "Social Ads", "Referrals", "Email"]
    cac = [42, 68, 124, 35, 22]
    ax4.barh(channels, cac, color="#ec4899", height=0.55, alpha=0.85)
    ax4.set_title("CAC by Channel ($)", fontsize=10, fontweight="bold", pad=8)
    ax4.grid(True, linestyle="--", alpha=0.2, color="#94a3b8")

    # Render figure to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img.resize((width, height), Image.Resampling.LANCZOS)


def generate_scientific_infographic(width=960, height=640) -> Image.Image:
    """Generates a publication-grade scientific data infographic."""
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="#ffffff")
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28, left=0.08, right=0.93, top=0.90, bottom=0.09)

    fig.text(0.08, 0.95, "Global Climate & Atmospheric Environmental Dynamics", fontsize=13, fontweight="bold", color="#1e293b")
    fig.text(0.08, 0.925, "Multi-variable empirical observation and statistical distribution analysis", fontsize=9, color="#64748b")

    # Subplot A: 2D Surface Temperature Anomaly Heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.linspace(-180, 180, 60)
    y = np.linspace(-90, 90, 40)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.radians(Y)) * np.cos(np.radians(X / 2)) + 0.4 * np.sin(np.radians(X * 2))
    cf = ax1.contourf(X, Y, Z, levels=14, cmap="coolwarm")
    cbar = fig.colorbar(cf, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Anomaly (deg C)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax1.set_title("(a) Surface Temperature Anomaly Map", fontsize=9, fontweight="bold", loc="left")
    ax1.set_xlabel("Longitude (deg)", fontsize=8)
    ax1.set_ylabel("Latitude (deg)", fontsize=8)
    ax1.tick_params(labelsize=7)

    # Subplot B: Atmospheric CO2 Trend & Prediction
    ax2 = fig.add_subplot(gs[0, 1])
    years = np.linspace(1980, 2026, 47)
    co2 = 338 + 1.8 * (years - 1980) + 0.015 * (years - 1980)**2 + np.random.RandomState(42).normal(0, 0.8, len(years))
    ax2.scatter(years, co2, color="#0284c7", s=16, alpha=0.7, label="Observed CO2 (ppm)")
    poly = np.polyfit(years, co2, 2)
    ax2.plot(years, np.polyval(poly, years), color="#dc2626", linewidth=2, label="Polynomial Fit (R2=0.99)")
    ax2.set_title("(b) Atmospheric CO2 Concentration Trend", fontsize=9, fontweight="bold", loc="left")
    ax2.set_xlabel("Year", fontsize=8)
    ax2.set_ylabel("Concentration (ppm)", fontsize=8)
    ax2.legend(loc="upper left", fontsize=7)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.tick_params(labelsize=7)

    # Subplot C: Polar Wind Velocity Streamfield / Vector field
    ax3 = fig.add_subplot(gs[1, 0])
    Y_grid, X_grid = np.mgrid[-5:5:20j, -5:5:20j]
    U = -1 - X_grid**2 + Y_grid
    V = 1 + X_grid - Y_grid**2
    speed = np.sqrt(U**2 + V**2)
    ax3.streamplot(X_grid, Y_grid, U, V, color=speed, cmap="plasma", linewidth=1.2, density=1.0)
    ax3.set_title("(c) Polar Vortical Flow Velocity Field", fontsize=9, fontweight="bold", loc="left")
    ax3.set_xlabel("Horizontal Axis (10^2 km)", fontsize=8)
    ax3.set_ylabel("Vertical Axis (10^2 km)", fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.3)
    ax3.tick_params(labelsize=7)

    # Subplot D: Multi-variable Distribution Boxplots
    ax4 = fig.add_subplot(gs[1, 1])
    np.random.seed(42)
    data = [
        np.random.normal(24.5, 3.2, 100),
        np.random.normal(18.2, 4.5, 100),
        np.random.normal(31.0, 2.8, 100),
        np.random.normal(12.4, 5.1, 100),
    ]
    bp = ax4.boxplot(data, patch_artist=True, tick_labels=["Zone A", "Zone B", "Zone C", "Zone D"],
                     medianprops=dict(color="#1e293b", linewidth=1.5))
    box_colors = ["#93c5fd", "#86efac", "#fde047", "#fca5a5"]
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_edgecolor("#475569")
    ax4.set_title("(d) Environmental Indicator Distributions", fontsize=9, fontweight="bold", loc="left")
    ax4.set_ylabel("Observed Metric Unit", fontsize=8)
    ax4.grid(True, linestyle="--", alpha=0.3)
    ax4.tick_params(labelsize=7)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img.resize((width, height), Image.Resampling.LANCZOS)


def generate_fallback_procedural(item: dict) -> Image.Image:
    """Generates rich procedural images matching semantic description when offline."""
    w, h = item["width"], item["height"]
    cat = item["category"]
    item_id = item["id"]

    if item_id == "sample_13_pcb_blueprint":
        return generate_pcb_blueprint(w, h)
    elif item_id == "sample_14_cad_blueprint":
        return generate_cad_blueprint(w, h)
    elif item_id == "sample_15_business_dashboard":
        return generate_business_dashboard(w, h)
    elif item_id == "sample_16_scientific_infographic":
        return generate_scientific_infographic(w, h)

    # General photographic / artistic procedural fallbacks
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)

    if item_id == "sample_01_wildlife_lion":
        # Savannah landscape gradient with lion silhouette and warm sunlight
        for y in range(h):
            r = int(240 - 0.2 * y)
            g = int(180 - 0.25 * y)
            b = int(100 - 0.15 * y)
            draw.line([(0, y), (w, y)], fill=(r, g, b))
        # Sun
        draw.ellipse([w // 2 - 80, 80, w // 2 + 80, 240], fill=(255, 230, 160))
        # Savannah grass
        draw.rectangle([0, int(h * 0.65), w, h], fill=(160, 120, 50))
        # Lion head & mane silhouette
        lx, ly = w // 2, int(h * 0.6)
        draw.ellipse([lx - 110, ly - 110, lx + 110, ly + 110], fill=(110, 70, 25))  # Mane
        draw.ellipse([lx - 70, ly - 60, lx + 70, ly + 60], fill=(200, 150, 70))    # Face
        draw.polygon([(lx - 55, ly - 80), (lx - 30, ly - 110), (lx - 15, ly - 75)], fill=(110, 70, 25))  # Left ear
        draw.polygon([(lx + 15, ly - 75), (lx + 30, ly - 110), (lx + 55, ly - 80)], fill=(110, 70, 25))  # Right ear
        draw.ellipse([lx - 35, ly - 15, lx - 15, ly + 5], fill=(240, 200, 80))    # Left eye
        draw.ellipse([lx + 15, ly - 15, lx + 35, ly + 5], fill=(240, 200, 80))    # Right eye
        draw.polygon([(lx - 15, ly + 25), (lx + 15, ly + 25), (lx, ly + 45)], fill=(40, 20, 10))  # Nose

    elif item_id == "sample_02_autumn_deer":
        # Misty forest gradient
        for y in range(h):
            t = y / h
            r = int(180 * (1 - t) + 120 * t)
            g = int(160 * (1 - t) + 100 * t)
            b = int(140 * (1 - t) + 70 * t)
            draw.line([(0, y), (w, y)], fill=(r, g, b))
        # Autumn tree trunks
        for tx in [120, 240, 580, 700]:
            draw.rectangle([tx, 0, tx + 28, h], fill=(80, 55, 35))
            draw.ellipse([tx - 50, 40, tx + 78, 200], fill=(210, 120, 30))  # Orange foliage
        # Deer with antlers
        dx, dy = 400, int(h * 0.65)
        draw.ellipse([dx - 50, dy - 30, dx + 50, dy + 30], fill=(140, 80, 40))   # Body
        draw.ellipse([dx + 30, dy - 60, dx + 70, dy - 15], fill=(150, 85, 45))   # Neck/Head
        # Antlers
        draw.line([(dx + 55, dy - 55), (dx + 80, dy - 110)], fill=(100, 60, 30), width=4)
        draw.line([(dx + 80, dy - 110), (dx + 110, dy - 130)], fill=(100, 60, 30), width=3)
        draw.line([(dx + 70, dy - 85), (dx + 95, dy - 95)], fill=(100, 60, 30), width=3)

    elif item_id == "sample_03_panoramic_mountain_lake":
        # 2:1 panoramic mountain lake
        for y in range(int(h * 0.55)):
            t = y / (h * 0.55)
            r = int(100 * (1 - t) + 210 * t)
            g = int(160 * (1 - t) + 230 * t)
            b = int(240 * (1 - t) + 255 * t)
            draw.line([(0, y), (w, y)], fill=(r, g, b))
        # Snow mountain peaks
        peaks = [(0, 300), (200, 100), (350, 220), (550, 60), (750, 200), (950, 80), (1200, 300)]
        draw.polygon(peaks + [(1200, 350), (0, 350)], fill=(235, 242, 250))
        # Lake reflection
        for y in range(int(h * 0.55), h):
            t = (y - h * 0.55) / (h * 0.45)
            r = int(20 * (1 - t) + 10 * t)
            g = int(100 * (1 - t) + 60 * t)
            b = int(150 * (1 - t) + 110 * t)
            draw.line([(0, y), (w, y)], fill=(r, g, b))

    elif item_id == "sample_04_panoramic_desert_dunes":
        # 2:1 panoramic golden desert dunes at sunset
        for y in range(h):
            t = y / h
            if t < 0.45:
                r = int(255 * (1 - t) + 240 * t)
                g = int(120 * (1 - t) + 160 * t)
                b = int(60 * (1 - t) + 80 * t)
            else:
                r = int(210 * (1 - t) + 160 * t)
                g = int(130 * (1 - t) + 90 * t)
                b = int(50 * (1 - t) + 30 * t)
            draw.line([(0, y), (w, y)], fill=(r, g, b))
        # Dunes curves
        dune_pts = [(0, 380), (300, 310), (650, 420), (950, 330), (1200, 400), (1200, h), (0, h)]
        draw.polygon(dune_pts, fill=(190, 110, 40))

    elif item_id == "sample_05_rainy_city_night":
        # 2:3 vertical rainy neon city
        for y in range(h):
            t = y / h
            draw.line([(0, y), (w, y)], fill=(int(10 + 20 * t), int(10 + 25 * t), int(25 + 40 * t)))
        # Skyscraper silhouettes with lit windows
        for sx, sw, sh in [(40, 160, 850), (220, 180, 920), (420, 150, 880), (580, 120, 800)]:
            draw.rectangle([sx, h - sh, sx + sw, h], fill=(15, 20, 35))
            for wy in range(h - sh + 40, h - 180, 32):
                for wx in range(sx + 15, sx + sw - 20, 24):
                    if (wx + wy) % 7 != 0:
                        draw.rectangle([wx, wy, wx + 12, wy + 16], fill=(255, 220, 120) if (wx + wy) % 3 == 0 else (80, 200, 255))
        # Wet road reflections
        draw.rectangle([0, h - 180, w, h], fill=(10, 12, 20))
        for rx, rc in [(120, (255, 40, 120)), (300, (40, 240, 255)), (500, (255, 200, 40))]:
            draw.ellipse([rx - 60, h - 140, rx + 60, h - 30], fill=rc)

    elif item_id == "sample_06_skyscraper_facade":
        # 2:3 vertical upward perspective of glass skyscrapers
        for y in range(h):
            draw.line([(0, y), (w, y)], fill=(int(80 - 0.05 * y), int(140 - 0.08 * y), int(220 - 0.1 * y)))
        # Converging glass facade diagonals
        draw.polygon([(0, 0), (220, h), (0, h)], fill=(40, 80, 130))
        draw.polygon([(w, 0), (w - 220, h), (w, h)], fill=(30, 70, 120))
        for gy in range(0, h, 40):
            draw.line([(0, gy), (w, gy)], fill=(120, 180, 240), width=1)

    elif item_id == "sample_07_macro_dewdrop":
        # 1:1 macro dewdrop on leaf with bokeh
        for y in range(h):
            t = y / h
            draw.line([(0, y), (w, y)], fill=(int(20 + 30 * t), int(80 + 70 * t), int(30 + 40 * t)))
        # Blurred background light circles (bokeh)
        for bx, by, br in [(120, 120, 60), (480, 180, 80), (160, 460, 70), (500, 480, 90)]:
            draw.ellipse([bx - br, by - br, bx + br, by + br], fill=(60, 160, 70))
        # Green leaf blade in foreground
        draw.polygon([(40, 320), (560, 260), (560, 380), (40, 440)], fill=(34, 139, 34))
        # Translucent dewdrop
        cx, cy, r = 300, 280, 65
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(160, 220, 230), outline=(255, 255, 255), width=3)
        draw.ellipse([cx - 25, cy - 35, cx - 10, cy - 20], fill=(255, 255, 255))  # Highlight

    elif item_id == "sample_08_macro_honeybee":
        # 1:1 macro sunflower disc with honeybee
        draw.rectangle([0, 0, w, h], fill=(220, 160, 20))  # Yellow petals bg
        draw.ellipse([100, 100, w - 100, h - 100], fill=(90, 50, 20))  # Sunflower seed center
        # Honeybee body
        bx, by = w // 2, h // 2
        draw.ellipse([bx - 60, by - 35, bx + 60, by + 35], fill=(220, 150, 30))
        # Black stripes
        for sx in [-30, 0, 30]:
            draw.rectangle([bx + sx - 8, by - 33, bx + sx + 8, by + 33], fill=(30, 20, 15))
        # Translucent wings
        draw.ellipse([bx - 40, by - 80, bx + 20, by - 15], fill=(230, 240, 255), outline=(180, 200, 220), width=2)
        draw.ellipse([bx - 10, by - 85, bx + 50, by - 20], fill=(230, 240, 255), outline=(180, 200, 220), width=2)

    elif item_id == "sample_09_impressionist_waterlilies":
        # Impressionist pastel pond with waterlilies
        for y in range(h):
            t = y / h
            draw.line([(0, y), (w, y)], fill=(int(40 + 60 * t), int(90 + 50 * t), int(110 - 20 * t)))
        # Water lily pads
        for px, py, pw, ph in [(200, 340, 140, 60), (450, 420, 180, 70), (680, 310, 150, 60), (180, 480, 160, 65)]:
            draw.ellipse([px - pw // 2, py - ph // 2, px + pw // 2, py + ph // 2], fill=(45, 120, 60))
            # Pink/White flowers
            draw.ellipse([px - 20, py - 20, px + 20, py + 10], fill=(255, 192, 203))
            draw.ellipse([px - 10, py - 12, px + 10, py + 5], fill=(255, 255, 255))

    elif item_id == "sample_10_starry_night":
        # Swirling cobalt blue & gold night sky
        for y in range(h):
            t = y / h
            draw.line([(0, y), (w, y)], fill=(int(15 + 20 * t), int(30 + 40 * t), int(90 + 30 * t)))
        # Swirling spirals
        for angle in np.linspace(0, 4 * math.pi, 200):
            r = angle * 22
            sx = int(450 + r * math.cos(angle))
            sy = int(240 + r * 0.4 * math.sin(angle))
            if 0 <= sx < w and 0 <= sy < h:
                draw.ellipse([sx - 6, sy - 6, sx + 6, sy + 6], fill=(240, 210, 60))
        # Dark cypress tree silhouette on left
        draw.polygon([(100, h), (140, 80), (180, h)], fill=(15, 25, 20))

    elif item_id == "sample_11_cyberpunk_anime":
        # Cyberpunk anime scene
        draw.rectangle([0, 0, w, h], fill=(16, 12, 36))
        # Window frame
        draw.rectangle([60, 60, w - 60, h - 60], outline=(0, 255, 240), width=4)
        # Glowing city skyline outside
        for sx in range(100, w - 100, 80):
            sh = 200 + (sx * 7) % 350
            draw.rectangle([sx, h - 80 - sh, sx + 65, h - 80], fill=(30, 24, 60), outline=(255, 40, 160), width=1)
        # Anime character silhouette
        draw.ellipse([w // 2 - 35, h // 2 - 120, w // 2 + 35, h // 2 - 50], fill=(255, 220, 200))
        draw.rectangle([w // 2 - 55, h // 2 - 50, w // 2 + 55, h - 100], fill=(20, 15, 30))

    elif item_id == "sample_12_fantasy_castle":
        # Fantasy floating island & castle
        for y in range(h):
            t = y / h
            draw.line([(0, y), (w, y)], fill=(int(120 + 100 * t), int(160 + 80 * t), int(240 - 20 * t)))
        # Floating island
        draw.ellipse([w // 2 - 220, h // 2, w // 2 + 220, h // 2 + 160], fill=(60, 130, 50))
        draw.polygon([(w // 2 - 200, h // 2 + 80), (w // 2 + 200, h // 2 + 80), (w // 2, h // 2 + 260)], fill=(90, 70, 50))
        # Castle towers
        for tx in [w // 2 - 100, w // 2, w // 2 + 100]:
            draw.rectangle([tx - 25, h // 2 - 140, tx + 25, h // 2], fill=(230, 235, 245), outline=(180, 190, 210), width=2)
            draw.polygon([(tx - 30, h // 2 - 140), (tx + 30, h // 2 - 140), (tx, h // 2 - 200)], fill=(60, 100, 180))

    return img


def build_diverse_dataset(output_dir: str, download: bool = True):
    """Builds all 16 diverse images and exports metadata JSON."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "diverse_dataset.json")

    print(f"============================================================")
    print(f" Building Diverse Multimodal Benchmark Dataset (16 Samples)")
    print(f" Target Directory: {output_dir}")
    print(f" Mode: {'Download with Procedural Fallback' if download else 'Offline Procedural Only'}")
    print(f"============================================================\n")

    metadata = {
        "version": "1.0",
        "description": "EmbeddingGemma-2 Diverse High-Quality Vision Benchmark",
        "samples": []
    }

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    for idx, item in enumerate(DIVERSE_DATASET_SPEC):
        file_path = os.path.join(output_dir, item["filename"])
        img = None

        print(f"[{idx + 1:02d}/16] Generating '{item['id']}' ({item['aspect_ratio']}, {item['width']}x{item['height']})...")

        # 0. Check if file already exists with high-resolution asset (e.g. curated digital illustration)
        if os.path.exists(file_path) and os.path.getsize(file_path) > 50000 and item["url"] is None:
            try:
                existing_img = Image.open(file_path).convert("RGB")
                if existing_img.size == (item["width"], item["height"]):
                    img = existing_img
                    print(f"       -> Reusing existing high-fidelity asset ({os.path.getsize(file_path):,} bytes).")
            except Exception:
                img = None

        # 1. Try download if URL is present and downloading is enabled
        if img is None and download and item["url"]:
            try:
                req = urllib.request.Request(item["url"], headers=headers)
                with urllib.request.urlopen(req, timeout=15) as response:
                    data = response.read()
                    downloaded_img = Image.open(io.BytesIO(data)).convert("RGB")
                    # Crop/Resize to exact target aspect ratio and resolution
                    img = downloaded_img.resize((item["width"], item["height"]), Image.Resampling.LANCZOS)
                    print(f"       -> Succeeded from CDN: {item['url'][:55]}...")
            except Exception as e:
                print(f"       -> CDN download failed ({e}), falling back to procedural generator.")

        # 2. Procedural generator fallback or primary source
        if img is None:
            img = generate_fallback_procedural(item)
            print(f"       -> Generated via High-Fidelity Procedural Generator.")

        # 3. Save as PNG
        img.save(file_path, "PNG", optimize=True)
        print(f"       -> Saved: {file_path} ({os.path.getsize(file_path):,} bytes)")

        # Record metadata
        entry = {
            "id": item["id"],
            "category": item["category"],
            "image_filename": item["filename"],
            "aspect_ratio": item["aspect_ratio"],
            "width": item["width"],
            "height": item["height"],
            "annotations": item["annotations"]
        }
        metadata["samples"].append(entry)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n============================================================")
    print(f" Dataset generation complete!")
    print(f" Total items: {len(metadata['samples'])}")
    print(f" Metadata saved to: {json_path}")
    print(f"============================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Build diverse vision benchmark dataset.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for diverse dataset (defaults to examples/embedding_mm/data/diverse)")
    parser.add_argument("--offline", action="store_true",
                        help="Do not download from web; generate all samples procedurally")
    args = parser.parse_args()

    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(os.path.dirname(script_dir), "data", "diverse")

    build_diverse_dataset(args.output_dir, download=not args.offline)


if __name__ == "__main__":
    main()
