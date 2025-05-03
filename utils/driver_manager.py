from utils import config
from utils.logger import Logger

from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logger = Logger().get_logger()
driver_path = ChromeDriverManager().install()


class MyService(Service):
    def __init__(self, executable_path: str, port: int = 0, service_args=None, log_path: str = None, env: dict = None):
        super(Service, self).__init__(executable_path, port, service_args, log_path, env, "Please see https://chromedriver.chromium.org/home")
        self.creationflags = 0x8000000


class DriverManager:
    @staticmethod
    def get_driver(headless=True, vpn=False):
        prefs = {"profile.default_content_setting_values": {"notifications": 2}}
        options = webdriver.ChromeOptions()
        options.add_experimental_option("prefs", prefs)
        if vpn:
            # https://chromewebstore.google.com/detail/vpn-free-betternet-unlimi/gjknjjomckknofjidppipffbpoekiipm
            options.add_extension(config.path["vpn_crx_path"])
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        options.add_argument("--log-level=3")
        if headless:
            options.add_argument("--headless=new")
            options.add_argument("--width=1920")
            options.add_argument("--height=1080")
        options.add_argument("--ignore-certificate-errors")
        options.add_argument("--allow-insecure-localhost")
        options.add_argument("--disable-web-security")
        # options.add_argument("--window-position=2000,0")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36")
        service = Service(driver_path)
        service.creation_flags = 0x8000000
        driver = webdriver.Chrome(service=service, options=options)

        driver.implicitly_wait(10)

        if vpn:
            check = 0
            while True:
                try:
                    driver.get("chrome-extension://gjknjjomckknofjidppipffbpoekiipm/panel/index.html")
                    _ = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, "//*[@id='screenMain']/div[3]/div[1]"))).click()
                    break
                except Exception as e:
                    check += 1
                    if check >= 4:
                        return None

        return driver
