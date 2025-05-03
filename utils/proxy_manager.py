import requests
import re
import pandas as pd
from queue import Queue
import warnings
from io import StringIO

warnings.filterwarnings("ignore")


class ProxyManager:
    def __init__(self):
        self.proxies_pool = Queue()
        self.proxies_list = []
        self.proxied_scraping()

    def proxied_scraping(self):
        response = requests.get("https://www.sslproxies.org/")
        proxy_df = pd.read_html(StringIO(response.text))
        proxy_ips = re.findall("\d+\.\d+\.\d+\.\d+:\d+", response.text)
        self.proxies_pool = Queue()  # Clear
        for proxy in proxy_ips:
            if not (proxy in self.proxies_list):
                self.proxies_pool.put(proxy)
                self.proxies_list.append(proxy)

    def get_proxy(self):
        if self.proxies_pool.qsize == 0:
            self.proxied_scraping()
        return self.proxies_pool.get()

    def return_proxy(self, proxy):
        return self.proxies_pool.put(proxy)

    def proxy_check(self, proxy):
        try:
            proxies = {"https": proxy}
            requests.get("https://httpbin.org/ip", proxies=proxies, timeout=20)
            return proxy
        except:
            return self.get_proxy()
