from bs4 import BeautifulSoup
import requests

url_text = requests.get('https://glints.com/id')
print(url_text)
