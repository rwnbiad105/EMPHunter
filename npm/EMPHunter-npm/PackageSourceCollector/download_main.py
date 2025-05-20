from web_crawler import rss_parser
from pkg_downloader import download_npm_in_list
from pkg_unpacker import unpackage_npm
import os
import re
import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def download_new_updated_npm_pkgs(date, limit):
    last_batch = get_batch_name(date)
    this_batch = last_batch + 1
    batch_name = date + "-" + str(this_batch)

    if os.path.exists(os.path.join("../DataShare/Result", batch_name)):
        print("Wrong Batch Name!")
        return
    rss_parser(date, limit, last_batch)
    print("rss_parser down")
    download_npm_in_list(batch_name)
    print("download_npm_in_list down")
    # unpackage_npm(batch_name)
    print("unpackage_npm down")

    return batch_name


def download_benign_pkgs(date, limit):
    last_batch = get_batch_name(date)
    this_batch = last_batch + 1
    batch_name = date + "-" + str(this_batch)
    if os.path.exists(os.path.join("../DataShare/BaseData", batch_name)):
        print("Wrong Batch Name!")
        return


def get_batch_name(date):
    batch_list = os.listdir(os.path.join(CURRENT_DIR, "../../Packages/packages"))
    last_batch = 0
    for batch_name in batch_list:
        if batch_name.startswith(date):
            pattern = r"^" + date + "-"
            # print(re.sub(pattern, "", batch_name))
            batch_num = int(re.sub(pattern, "", batch_name))
            if batch_num > last_batch:
                last_batch = batch_num
    print(date + "-" + str(last_batch + 1))
    return last_batch


if __name__ == "__main__":
    # batch_name = "20240301-1"
    date = str(datetime.date.today())
    date = "".join(date.split("-"))
    limit = 500
    download_new_updated_npm_pkgs(date, limit)
    # https://registry.npmjs.org/-/rss?descending=true&limit=600