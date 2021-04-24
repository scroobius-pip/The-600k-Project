import redis
import itertools
from PIL import Image, UnidentifiedImageError
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


from pickle import loads, dumps, dump, load
from pympler import muppy, summary

from redis_store import RedisStore

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")


THREAD_COUNT = 10000


def load_current_index():
    try:
        return load(open("current_index.p", "rb"))
    except:
        save_current_index(0)
        return 0


def save_current_index(current_index):
    with open("current_index.p", "wb") as f:
        dump(current_index, f)


redis_start_index = load_current_index()
failed = 0
current_index = redis_start_index
print(f'current index is {current_index}')


def set_interval(func, sec):
    def func_wrapper():
        set_interval(func, sec)
        func()
    t = threading.Timer(sec, func_wrapper)
    t.start()
    return t


def print_dump():
    all_objects = muppy.get_objects()
    sum1 = summary.summarize(all_objects)
    # Prints out a summary of the large objects
    summary.print_(sum1)


def store_processed_image(user_image_hash_str, user_image_hash, user_image):
    # db[user_image_hash_str] = (user_image, user_image_hash)
    pass


def append_twitter_profile_img_jpg(profile_img):
    return [f'https://pbs.twimg.com/profile_images/{profile_img}_200x200.jpg', f'https://pbs.twimg.com/profile_images/{profile_img}_200x200.jpeg']


def append_twitter_profile_img_png(profile_img):
    return [f'https://pbs.twimg.com/profile_images/{profile_img}_200x200.png']


def get_png_from_profile_img(profile_img):
    urls = append_twitter_profile_img_png(profile_img)
    for url in urls:
        try:
            response = requests.get(url)

            png = Image.open(BytesIO(response.content)).convert('RGBA')
            background = Image.new('RGBA', png.size, (255, 255, 255))

            alpha_composite = Image.alpha_composite(background, png)
            resized_image = alpha_composite.resize((50, 50))

            converted_image = resized_image.convert('RGB')

            return converted_image
        except Exception as exception:
            # print(f'{type(exception).__name__} {url}')
            continue

    raise UnidentifiedImageError()


def get_jpg_from_profile_img(profile_img):
    urls = append_twitter_profile_img_jpg(profile_img)
    for url in urls:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            resized_image = image.resize((50, 50))
            image.close()
            return resized_image
        except Exception as exception:
            # print(f'{type(exception).__name__} {url}')
            continue

    raise UnidentifiedImageError()


image_functions = [get_jpg_from_profile_img, get_png_from_profile_img]
formats = ["jpg", "png"]


def get_image_from_profile(profile):
    # print(url)
    (user_name, profile_img) = profile
    global failed
    for i, code in enumerate(image_functions):
        try:
            image = code(profile_img)
            # logging.info(f'success {profile_img} {formats[i]}')
            return image
        except UnidentifiedImageError as exception:
            # logging.error(
            #     f'Could not get {formats[i]} for {profile_img} {type(exception).__name__}')
            continue
        except Exception as exception:
            logging.error(
                f'{type(exception).__name__} {profile_img} {user_name}')
            break

    failed += 1
    # logging.error(f'failed {profile_img} {user_name}')
    return None


def get_process_image(profile: str):
    user_image = get_image_from_profile(profile)
    if user_image is not None:
        user_image_hash = imagehash.colorhash(user_image)
        userimages_hash[str(user_image_hash)] = (user_image, user_image_hash)

    # logging.info(
    #     f'Thread finished, userimages_hash contains {len(userimages_hash)} profile urls')
    # print_dump()


tic = time.perf_counter()


userimages_hash = RedisStore(1)
r = redis.Redis(host='ec2-3-236-123-111.compute-1.amazonaws.com')

user_generator = r.sscan_iter('twitterusers')
first_thousand_users = itertools.islice(user_generator, current_index, 6000000)

set_interval(lambda: save_current_index(current_index), 60)

with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
    for index, users_byte_string in enumerate(first_thousand_users):
        # print(index)
        try:
            current_index = index+redis_start_index
            [user_name, profile_img, _] = str(
                users_byte_string, 'utf-8').split('\n')
            # print(str(users_byte_string), profile_img)
            executor.submit(get_process_image, (user_name, profile_img))
        except Exception as exception:
            logging.error(f'{type(exception).__name__} {users_byte_string}')

logging.info(f'failed {failed}')
# print('serializing to disk')
# compressed_pickle('userimages_hash_all', userimages_hash)

toc = time.perf_counter()
logging.info(f'finished in {toc - tic:0.4f} seconds')
