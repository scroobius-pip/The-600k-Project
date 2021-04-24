import redis
import itertools
from PIL import Image
import requests
import imagehash
import vptree
from io import BytesIO
import pickle
import logging
import threading
import time
import concurrent.futures
import bz2
import pickle
from redis_store import RedisStore

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

neighbour_cache = {}
neighbours_cache = {}
THREAD_COUNT = 1000


def set_interval(func, sec):
    def func_wrapper():
        set_interval(func, sec)
        func()
    t = threading.Timer(sec, func_wrapper)
    t.start()
    return t


def img_hash_distance(a, b):
    return a-b


def section_image_fun(image: Image.Image, size=5):
    width = image.width
    height = image.height
    sections = [[image.crop((x, y, x+size, y+size)) for x in range(0, width+1, size)]
                for y in range(0, height+1, size)]
    return sections


def serialize_cache(neighbour_cache, neighbours_cache):
    try:
        print(f'neighbour_contains {len(neighbour_cache)}')
        print(f'neighbours_contains {len(neighbours_cache)}')
        compressed_pickle('cache', (neighbour_cache, neighbours_cache))
        # compressed_pickle('neighbour_cache', neighbour_cache)
        # compressed_pickle('neighbours_cache', neighbours_cache)
        return True
    except:
        print('failed to serialize cache')
        return False


def deserialize_cache():
    try:
        return decompress_pickle('cache')
    except:
        print('failed to deserialize cache')
        return ({}, {})


def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        pickle.dump(data, f)


def decompress_pickle(title):
    with bz2.BZ2File(title + '.pbz2', 'r') as f:
        return pickle.load(f)


def update_new_image(sectioned_image, initial_tree, new_image, userimages_hash, x, y):
    section_image = sectioned_image[x][y]
    section_hash = imagehash.colorhash(section_image)
    section_hash_str = str(section_hash)

    neighbour = None

    if section_hash_str in neighbour_cache:
        neighbour = neighbour_cache[section_hash_str]
    else:
        neighbour = initial_tree.get_nearest_neighbor(section_hash)
        neighbour_cache[section_hash_str] = neighbour

    (_, closest_hash) = neighbour
    closest_hash_str = str(closest_hash)
    (closest_image, _) = userimages_hash[closest_hash_str]
    new_image.paste(
        closest_image, (y*closest_image.height, x*closest_image.width))


def update_new_image_neighbours(sectioned_image, initial_tree, new_image, hash_counter, userimages_hash, max_repeat, neighbour_count, x, y):
    section_image = sectioned_image[x][y]
    section_hash = imagehash.colorhash(section_image)

    section_hash_str = str(section_hash)

    neighbours = None

    if section_hash_str in neighbours_cache:
        neighbours = neighbours_cache[section_hash_str]
    else:
        neighbours = initial_tree.get_n_nearest_neighbors(
            section_hash, neighbour_count)
        neighbours_cache[section_hash_str] = neighbours

    # sorted_neighbours = sorted(neighbours, key=lambda tup: tup[0])
    for neighbour in neighbours:
        (_, closest_hash) = neighbour
        closest_hash_str = str(closest_hash)
        if closest_hash_str in hash_counter:
            if(hash_counter[closest_hash_str] > max_repeat):
                continue

        (closest_image, _) = userimages_hash[closest_hash_str]
        hash_counter[closest_hash_str] = hash_counter.get(
            closest_hash_str, 0) + 1
        new_image.paste(
            closest_image, (y*closest_image.height, x*closest_image.width))
        break


def create_collage(scale, host_img, job_dict, initial_tree, userimages_hash):
    section_scale = scale["section_scale"]
    image_scale = scale["image_scale"]
    output_name = job_dict["output_name"]
    no_repeat = job_dict["no_repeat"]
    # outputfile_name = f'{output_name}_{section_scale}_{image_scale}.jpg' if no_repeat is False else f'{output_name}_{section_scale}_{image_scale}_{job_dict["neighbour_count"]_{job_dict["max_repeat"]}}.jpg'
    outputfile_name = '_'.join([str(x) for x in [output_name, section_scale, image_scale, job_dict.get(
        "neighbour_count"), job_dict.get("max_repeat")] if x is not None])+'.jpg'
    logging.info(f'started collage {outputfile_name}')
    tic = time.perf_counter()
    downsized_host_img = host_img.resize(
        (host_img.width//image_scale, host_img.height//image_scale))
    sectioned_image = section_image_fun(downsized_host_img, section_scale)
    new_image = Image.new(
        "RGB", (len(sectioned_image[0])*50, len(sectioned_image)*50))

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        if no_repeat is True:
            hash_counter = {}
            for x in range(len(sectioned_image)):
                for y in range(len(sectioned_image[x])):
                    executor.submit(update_new_image_neighbours, sectioned_image, initial_tree, new_image,
                                    hash_counter, userimages_hash, job_dict["max_repeat"], job_dict["neighbour_count"], x, y)
                    # update_new_image_neighbours(
                    #     sectioned_image, initial_tree, new_image, hash_counter, userimages_hash, job_dict["max_repeat"], job_dict["neighbour_count"], x, y)

        else:
            for x in range(len(sectioned_image)):
                for y in range(len(sectioned_image[x])):
                    executor.submit(update_new_image, sectioned_image,
                                    initial_tree, new_image, userimages_hash, x, y)

    new_image.save(f'./{outputfile_name}')
    toc = time.perf_counter()
    logging.info(f"created {outputfile_name} in {toc - tic:0.4f} seconds\n")


def create_job(job_dict, initial_tree, userimages_hash):
    logging.info(f'started job {job_dict["name"]}')
    tic = time.perf_counter()
    with Image.open(job_dict["input_file"]) as host_img:
        with concurrent.futures.ThreadPoolExecutor(max_workers=job_dict["thread_count"]) as executor:
            for scale in job_dict["scales"]:
                executor.submit(create_collage, scale, host_img, job_dict,
                                initial_tree, userimages_hash)
                # create_collage(scale, host_img, job_dict,
                #             initial_tree, userimages_hash)

    toc = time.perf_counter()
    logging.info(
        f'finished job {job_dict["name"]} in {toc - tic:0.4f} seconds ')


#
#
#
#
#
#
#
#
#
#
#
jobs = [

  
    # {
    #     "no_repeat": True,
    #     "max_repeat": 10,
    #     "neighbour_count": 1000,
    #     "thread_count": 2,
    #     "name": "lain",
    #     "input_file": "lain.jpg",
    #     "output_name": "lain",

    #     "scales": [

    #         {"section_scale": 1, "image_scale": 1},
    #         {"section_scale": 2, "image_scale": 1},
    #         # {"section_scale": 2, "image_scale": 2},

    #     ]
    # },
   

]

big_jobs = [



    # {
    #     "no_repeat": True,
    #     "max_repeat": 100,
    #     "neighbour_count": 2000,
    #     "thread_count": 1,
    #     "name": "host1",
    #     "input_file": "host1.jpg",
    #     "output_name": "collage1_v2",

    #     "scales": [

    #         {"section_scale": 1, "image_scale": 2},
    #         {"section_scale": 1, "image_scale": 1},

    #         {"section_scale": 2, "image_scale": 2},
    #     ]
    # },
    {
        "no_repeat": False,
        # "max_repeat": 100,
        # "neighbour_count": 2000,
        "thread_count": 1,
        "name": "host1",
        "input_file": "host1.jpg",
        "output_name": "collage1_v3",

        "scales": [

            {"section_scale": 1, "image_scale": 1},

            # {"section_scale": 2, "image_scale": 2},
        ]
    },
    # {
    #     "no_repeat": True,
    #     "max_repeat": 100,
    #     "neighbour_count": 2000,
    #     "thread_count": 1,
    #     "name": "flowers",
    #     "input_file": "flowers.jpg",
    #     "output_name": "flowers",

    #     "scales": [

    #         {"section_scale": 1, "image_scale": 2},

    #         {"section_scale": 2, "image_scale": 2},
    #     ]
    # },
    # {
    #     "no_repeat": False,
    #     # "max_repeat": 100,
    #     # "neighbour_count": 2000,
    #     "thread_count": 1,
    #     "name": "flowers",
    #     "input_file": "flowers.jpg",
    #     "output_name": "flowers",

    #     "scales": [

    #         {"section_scale": 1, "image_scale": 2},

    #         {"section_scale": 2, "image_scale": 2},
    #     ]
    # },
    # {
    #     "no_repeat": True,
    #     "max_repeat": 100,
    #     "neighbour_count": 2000,
    #     "thread_count": 1,
    #     "name": "host2",
    #     "input_file": "host2.jpg",
    #     "output_name": "collage2_v2",

    #     "scales": [

    #         {"section_scale": 1, "image_scale": 2},

    #         {"section_scale": 2, "image_scale": 2},
    #     ]
    # },
    # {
    #     "no_repeat": True,
    #     "max_repeat": 100,
    #     "neighbour_count": 2000,
    #     "thread_count": 1,
    #     "name": "pattern",
    #     "input_file": "pattern.jpg",
    #     "output_name": "pattern",

    #     "scales": [

    #         {"section_scale": 1, "image_scale": 2},

    #         {"section_scale": 2, "image_scale": 2},
    #     ]
    # },
    # {
    #     "no_repeat": True,
    #     "max_repeat": 100,
    #     "neighbour_count": 2000,
    #     "thread_count": 1,
    #     "name": "woman",
    #     "input_file": "woman.jpg",
    #     "output_name": "woman",

    #     "scales": [

    #         {"section_scale": 1, "image_scale": 2},
    #         {"section_scale": 1, "image_scale": 1},
    #         {"section_scale": 2, "image_scale": 2},

    #     ]
    # },
    # {
    #     "no_repeat": True,
    #     "max_repeat": 100,
    #     "neighbour_count": 2000,
    #     "thread_count": 1,
    #     "name": "space",
    #     "input_file": "space.jpg",
    #     "output_name": "space",

    #     "scales": [

    #         {"section_scale": 1, "image_scale": 2},
    #         {"section_scale": 1, "image_scale": 4},
    #         {"section_scale": 1, "image_scale": 1},

    #     ]
    # },
    # {
    #     "no_repeat": True,
    #     "max_repeat": 100,
    #     "neighbour_count": 2000,
    #     "thread_count": 1,
    #     "name": "evangelion",
    #     "input_file": "evangelion.png",
    #     "output_name": "evangelion",

    #     "scales": [

    #         {"section_scale": 1, "image_scale": 2},

    #     ]
    # },
    # {
    #     "no_repeat": False,
    #     # "max_repeat": 100,
    #     # "neighbour_count": 2000,
    #     "thread_count": 1,
    #     "name": "evangelion",
    #     "input_file": "evangelion.png",
    #     "output_name": "evangelion",

    #     "scales": [

    #         {"section_scale": 1, "image_scale": 2},
    #         {"section_scale": 2, "image_scale": 2},

    #     ]
    # },



]


tic = time.perf_counter()

print('deserializing userimage_hash and cache files')
# userimages_hash = decompress_pickle('userimages_hash')
userimages_hash = {}
redis_store = RedisStore(1)
for key_byte in redis_store:
    key_str = str(key_byte, 'utf-8')

    userimages_hash[key_str] = redis_store.get(key_byte)

cache_result = deserialize_cache()
neighbour_cache = cache_result[0]
neighbours_cache = cache_result[1]
set_interval(lambda: serialize_cache(neighbour_cache, neighbours_cache), 60)
print(f'userimages_hash size: {len(userimages_hash)}')
print(f'neighbour_cache size: {len(neighbour_cache)}')
print(f'neighbours_cache size: {len(neighbours_cache)}')

print(f"deserialized in {time.perf_counter() - tic:0.4f} seconds ")

initial_tree = vptree.VPTree(
    list(map(lambda tu: tu[1], userimages_hash.values())), img_hash_distance)

print(f"starting jobs")

tic2 = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    for job in jobs:
        executor.submit(create_job, job, initial_tree, userimages_hash)
print(
    f"finished {len(jobs)} small jobs in {time.perf_counter() - tic2:0.4f} seconds ")

tic3 = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    for job in big_jobs:
        # executor.submit(create_job, job, initial_tree, userimages_hash)
        create_job(job, initial_tree, userimages_hash)


print('serializing caches')
serialize_cache(neighbour_cache, neighbours_cache)
# print('serializing caches')

print(
    f"finished {len(big_jobs)} big jobs in {time.perf_counter() - tic3:0.4f} seconds ")
