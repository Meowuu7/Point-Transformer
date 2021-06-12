#_*_ coding:utf-8 _*_
import os
import requests
import time
import random
from lxml import etree
from urllib.request import urlopen

from bs4 import BeautifulSoup
from PIL import Image
# from cStringIO import StringIO
import urllib
from urllib.request import urlretrieve
keyWord = "aa"
#  = input(f"{'Please input the keywords that you want to download :'}")
class Spider():
    #初始化参数
    def __init__(self):
        self.headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.104 Safari/537.36",
        }
        self.filePath = ('../tox_imgs')
        if not os.path.exists(self.filePath):
            os.mkdir(self.filePath)
        self.general_names = {} #"edge_texture" ,"edge_occlusion", "reshading", "segment_semantic", "rgb"
        self.img_type_of_interest = ["principal_curvature"]
        self.fn_type = "albertville_normal"
        self.root_dir = "../XTConsistency/tools/data"

    def creat_File(self):
        filePath = self.filePath
        if not os.path.exists(filePath):
            os.makedirs(filePath)

    def get_pageNum(self):
        html = urlopen("https://github.com/Meowuu7/taskonomy-sample-model-1/tree/master/rgb")
        bs = BeautifulSoup(html.read(), features="lxml")
        # print(bs)
        img_as = bs.findAll("a", {"class": "js-navigation-open Link--primary"})
        print(len(img_as))
        print(img_as[0])
        print(img_as[0].string)
        raw_strings = []
        root_dir = "../XTConsistency/tools/data"
        for img_a in img_as:
            img_a_string = img_a.string
            general_name = "_".join(img_a_string.split("_")[:-1])
            raw_strings.append(general_name)
        with open(os.path.join(root_dir, "general_names.txt"), "w") as wf:
            for ss in raw_strings:
                wf.write(ss + "\n")
            wf.close()
        print("writen!")
        # img_divs = bs.findAll("div", {"class": "Box-row Box-row--focus-gray py-2 d-flex position-relative js-navigation-item "})
        # print(len(img_divs))
        # links = []
        # for img_div in img_divs:
        #     rowh = img_div.findAll("div", {"class": "flex-auto min-width-0 col-md-2 mr-3"})[0].findAll("span", {"class": "css-truncate css-truncate-target d-block width-fit"})
        #     a = rowh.findAll("a")[0]
        #     lin = a['href']
        #     links.append(lin)
        # print(len(links))
        # print(links[0])
        #
        # totalPagenum = len(links)
        # self.links = links
        # return totalPagenum

    def get_all_file_names(self):
        root_dir = "../XTConsistency/tools/data"
        an = []
        with open(os.path.join(root_dir, "general_names.txt"), "r") as rf:
            for line in rf:
                an.append(line.strip())
        self.all_names = an

    def get_interest_file_name(self, fn_ty):
        # fn_ty \in ["albertville_normal", "almena_normal"]
        root_dir = "../XTConsistency/tools/data"
        file_names = []
        file_dir = os.path.join(root_dir, fn_ty)
        file_dir = os.path.join(file_dir, fn_ty.split("_")[-1])
        print("Seeking files in %s" % file_dir)
        file_names = os.listdir(file_dir)
        general_names = ["_".join(fn.split("_")[:-1]) for fn in file_names]
        self.general_names[fn_ty.split("_")[0]] = general_names
        print("Got general names with total length = %d" % len(general_names))

    # img_types_of_interest = ["edge_occlusion", ]
    def _get_imgs_for_one_type(self, ty_name):
        assert ty_name in self.img_type_of_interest
        general_names = self.general_names[self.fn_type.split("_")[0]]
        img_folder_nm = os.path.join(self.root_dir, self.fn_type.split("_")[0] + "_" + ty_name, ty_name)
        if not os.path.exists(img_folder_nm):
            os.mkdir("/".join(img_folder_nm.split("/")[:-1]))
            os.mkdir(img_folder_nm)

        for i, gn in enumerate(general_names):
            if ty_name == "segment_semantic":
                img_name = gn + "_" + "segmentsemantic" + ".png"
            else:
                img_name = gn + "_" + ty_name + ".png"
            pic_path = os.path.join(img_folder_nm, img_name)
            url = "https://raw.githubusercontent.com/Meowuu7/taskonomy-sample-model-1/master/" + ty_name + "/" + img_name
            pic = requests.get(url, headers=self.headers)
            f = open(pic_path, 'wb')
            f.write(pic.content)
            f.close()

    def _get_remained_imgs_for_one_type(self, ty_name):
        assert ty_name in self.img_type_of_interest
        general_names = self.general_names[self.fn_type.split("_")[0]]
        general_names_remained = [nn for nn in self.all_names if nn not in general_names]
        img_folder_nm = os.path.join(self.root_dir, self.fn_type.split("_")[0] + "_" + ty_name, ty_name)
        if not os.path.exists(img_folder_nm):
            os.mkdir("/".join(img_folder_nm.split("/")[:-1]))
            os.mkdir(img_folder_nm)
        else:
            ext_file_nms = os.listdir(img_folder_nm)
            ext_general_nms = ["_".join(fn.split("_")[:5]) for fn in ext_file_nms]
            general_names_remained = [fn for fn in general_names_remained if fn not in ext_general_nms]
        for i, gn in enumerate(general_names_remained):
            if ty_name == "segment_semantic":
                img_name = gn + "_" + "segmentsemantic" + ".png"
            else:
                img_name = gn + "_" + ty_name + ".png"
            pic_path = os.path.join(img_folder_nm, img_name)
            url = "https://raw.githubusercontent.com/Meowuu7/taskonomy-sample-model-1/master/" + ty_name + "/" + img_name
            pic = requests.get(url, headers=self.headers)
            f = open(pic_path, 'wb')
            f.write(pic.content)
            f.close()

    def get_imgs_for_all_types(self):
        for ty_name in self.img_type_of_interest:
            print("Start getting imgs of type %s" % ty_name)
            self._get_imgs_for_one_type(ty_name)
            print("Got them!")

    def get_remained_imgs_for_all_types(self):
        for ty_name in self.img_type_of_interest:
            print("Start getting imgs of type %s" % ty_name)
            self._get_remained_imgs_for_one_type(ty_name)
            print("Got them!")



    def main_fuction(self):
        self.creat_File()
        count = self.get_pageNum()
        print("We have found:{} images!".format(count))
        times = int(count/24 + 1)
        j = 1
        for i in range(times):
            pic_Urls = self.getLinks(i+1)
            for item in pic_Urls:
                self.download(item,j)
                j += 1

    def getLinks(self,number):
        url = ("https://alpha.wallhaven.cc/search?q={}&categories=111&purity=100&sorting=relevance&order=desc&page={}").format(keyWord,number)
        try:
            html = requests.get(url)
            selector = etree.HTML(html.text)
            pic_Linklist = selector.xpath('//a[@class="jsAnchor thumb-tags-toggle tagged"]/@href')
        except Exception as e:
            print(repr(e))
        return pic_Linklist

    def download(self,url,count):
        for lin in self.links:
            pic_path = os.path.join(self.filePath, lin.split("/")[-1])
            html = "https://github.com/" + lin
            pic = requests.get(html, headers=self.headers)
            f = open(pic_path, 'wb')
            f.write(pic.content)
            f.close()

        # string = url.strip('/thumbTags').strip('https://alpha.wallhaven.cc/wallpaper/')
        # html = 'http://wallpapers.wallhaven.cc/wallpapers/full/wallhaven-' + string + '.jpg'
        # pic_path = (self.filePath + keyWord + str(count) + '.jpg' )
        # try:
        #     pic = requests.get(html,headers = self.headers)
        #
        #     print("Image:{} has been downloaded!".format(count))
        #     time.sleep(random.uniform(0,2))
        # except Exception as e:
        #     print(repr(e))


if __name__=='__main__':
    spider = Spider()
    # spider.get_interest_file_name("albertville_normal")
    # # spider.get_imgs_for_all_types()
    # spider._get_imgs_for_one_type("segment_semantic")
    spider.get_pageNum()
    spider.get_interest_file_name("albertville_normal")
    spider.get_all_file_names()
    spider.get_remained_imgs_for_all_types()
