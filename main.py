import requests
from bs4 import BeautifulSoup


def get_article(url):
    page = requests.get(url)

    soup = BeautifulSoup(page.content, 'html.parser')
    body = soup.find_all(class_='cnnBodyText')[2].get_text()

    with open("news.txt", 'w') as f:
        f.write(body)


if __name__ == '__main__':
    # make list of urls, iterate through each url calling get_article[x] and writing to file a new file each time with different name
    get_article('https://transcripts.cnn.com/show/cnr/date/2021-10-03/segment/06')


