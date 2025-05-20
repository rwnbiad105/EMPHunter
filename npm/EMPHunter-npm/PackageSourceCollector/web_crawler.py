import feedparser
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def rss_parser(date, size=600, last_batch=1):
    batch_name = date + "-" + str(last_batch + 1)
    last_batch_name = date + "-" + str(last_batch)
    if not os.path.exists(CURRENT_DIR + "/../DataShare"):
        os.mkdir(CURRENT_DIR + "/../DataShare")
    if not os.path.exists(CURRENT_DIR + "/../../Packages/packages"):
        os.mkdir(CURRENT_DIR + "/../../Packages/packages")
    if not os.path.exists(os.path.join(CURRENT_DIR + "/../../Packages/packages", batch_name)):
        os.mkdir(os.path.join(CURRENT_DIR + "/../../Packages/packages", batch_name))

    print("start to parse rss")
    rss_content = feedparser.parse("https://registry.npmjs.org/-/rss?descending=true&limit=" + str(size))
    print(len(rss_content))

    with open(os.path.join(CURRENT_DIR, "../../Packages/packages", batch_name, batch_name + "_names.txt"), "w+") as f:
        last_name_list = []
        if os.path.exists(os.path.join(CURRENT_DIR, "../../Packages/packages", last_batch_name, last_batch_name + "_names.txt")):
            print("get last name list")
            with open(os.path.join(CURRENT_DIR, "../../Packages/packages", last_batch_name, last_batch_name + "_names.txt"), "r") as lf:
                temp_list = lf.readlines()
            for name in temp_list:
                last_name_list.append(name.strip())

        for name in rss_content.__getitem__("entries"):
            if name.title not in last_name_list:
                f.write(name.title + "\n")


if __name__ == "__main__":
    rss_parser("20240121-3", 600)