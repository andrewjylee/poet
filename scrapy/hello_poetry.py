import scrapy

class HelloPoetry(scrapy.Spider):
    name = 'hello_poetry'
    download_delay = 1.5

    def start_requests(self):
        base_url = 'http://hellopoetry.com/%s/poems/title/?page=' % 'sylvia-plath'
        urls = [base_url + str(x) for x in range(1, 8)]

        base_url = 'http://hellopoetry.com/%s/poems/title/?page=' % 'oscar-wilde'
        urls.extend([base_url + str(x) for x in range(1, 6)])

        base_url = 'http://hellopoetry.com/%s/poems/title/?page=' % 'e-e-cummings'
        urls.extend([base_url + str(x) for x in range(1, 8)])

        base_url = 'http://hellopoetry.com/%s/poems/title/?page=' % 'ernest-hemingway-1'
        urls.extend([base_url + str(x) for x in range(1, 3)])

        base_url = 'http://hellopoetry.com/%s/poems/title/?page=' % 'rumi'
        urls.extend([base_url + str(x) for x in range(1, 3)])

        base_url = 'http://hellopoetry.com/%s/poems/title/?page=' % 'bob-dylan-robert-zimmerman'
        urls.extend([base_url])


        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse) 

    def parse(self, response):
        for text in response.xpath('//div[contains(@class, "poem-part continue-reading poem-body wordwrap")]/p/text()'):
            print text.extract().encode('utf-8')
