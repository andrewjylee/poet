import scrapy

class QuotesSpider(scrapy.Spider):
    name = 'poems'

    def start_requests(self):
        urls = []
        urls.extend(['http://famouspoetsandpoems.com/poets/emily_dickinson/poems/'+str(x) for x in range(5212, 11259)])
        urls.extend(['http://famouspoetsandpoems.com/poets/pablo_neruda/poems/'+str(x) for x in range(15703, 15747)])
        urls.extend(['http://famouspoetsandpoems.com/poets/walt_whitman/poems/'+str(x) for x in range(17463, 17792)])
        urls.extend(['http://famouspoetsandpoems.com/poets/william_carlos_williams/poems/'+str(x) for x in range(16997, 17055)])
        urls.extend(['http://famouspoetsandpoems.com/poets/t__s__eliot/poems/'+str(x) for x in range(15120, 15199)])
        urls.extend(['http://famouspoetsandpoems.com/poets/ezra_pound/poems/'+str(x) for x in range(18774, 18845)])
        urls.extend(['http://famouspoetsandpoems.com/poets/ralph_waldo_emerson/poems/'+str(x) for x in range(15262, 15359)])
        urls.extend(['http://famouspoetsandpoems.com/poets/wallace_stevens/poems/'+str(x) for x in range(18024, 18057)])
        urls.extend(['http://famouspoetsandpoems.com/poets/marianne_moore/poems/'+str(x) for x in range(15525, 15543)])
        urls.extend(['http://famouspoetsandpoems.com/poets/gertrude_stein/poems/'+str(x) for x in range(18081, 18084)])
        urls.extend(['http://famouspoetsandpoems.com/poets/hart_crane/poems/'+str(x) for x in range(11355, 11368)])
        urls.extend(['http://famouspoetsandpoems.com/poets/robert_frost/poems/'+str(x) for x in range(528, 810)])
        urls.extend(['http://famouspoetsandpoems.com/poets/langston_hughes/poems/'+str(x) for x in range(16944, 16973)])
        urls.extend(['http://famouspoetsandpoems.com/poets/dylan_thomas/poems/'+str(x) for x in range(11394, 11457)])
        urls.extend(['http://famouspoetsandpoems.com/poets/john_keats/poems/'+str(x) for x in range(14372, 14456)])
        urls.extend(['http://famouspoetsandpoems.com/poets/ted_hughes/poems/'+str(x) for x in range(13792, 13937)])
        urls.extend(['http://famouspoetsandpoems.com/poets/thomas_hardy/poems/'+str(x) for x in range(10687, 11678)])
        urls.extend(['http://famouspoetsandpoems.com/poets/shel_silverstein/poems/'+str(x) for x in range(14818, 14837)])
        urls.extend(['http://famouspoetsandpoems.com/poets/carl_sandburg/poems/'+str(x) for x in range(843, 3235)])
        urls.extend(['http://famouspoetsandpoems.com/poets/rudyard_kipling/poems/'+str(x) for x in range(14456, 20918)])
        urls.extend(['http://famouspoetsandpoems.com/poets/nazim_hikmet/poems/'+str(x) for x in range(13748, 13769)])
        urls.extend(['http://famouspoetsandpoems.com/poets/jack_prelutsky/poems/'+str(x) for x in range(18767, 18774)])
        urls.extend(['http://famouspoetsandpoems.com/poets/edgar_allan_poe/poems/'+str(x) for x in range(18847, 18869)])
        urls.extend(['http://famouspoetsandpoems.com/poets/anna_akhmatova/poems/'+str(x) for x in range(12, 39)])
        urls.extend(['http://famouspoetsandpoems.com/poets/lewis_carroll/poems/'+str(x) for x in range(6085, 6433)])
        urls.extend(['http://famouspoetsandpoems.com/poets/james_joyce/poems/'+str(x) for x in range(2440, 2491)])
        urls.extend(['http://famouspoetsandpoems.com/poets/walter_de_la_mare/poems/'+str(x) for x in range(1914, 1961)])
        urls.extend(['http://famouspoetsandpoems.com/poets/henry_david_thoreau/poems/'+str(x) for x in range(179214, 17943)])
        urls.extend(['http://famouspoetsandpoems.com/poets/maya_angelou/poems/'+str(x) for x in range(482, 513)])
        urls.extend(['http://famouspoetsandpoems.com/poets/william_butler_yeats/poems/'+str(x) for x in range(10173, 10915)])
        urls.extend(['http://famouspoetsandpoems.com/poets/charles_bukowski/poems/'+str(x) for x in range(12976, 13266)])
        urls.extend(['http://famouspoetsandpoems.com/poets/theodore_roethke/poems/'+str(x) for x in range(16314, 16336)])

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for text in response.xpath('//div[contains(@style, "padding-left:14px;padding-top:20px;font-family:Arial;font-size:13px")]/text()'):
            print text.extract().encode('utf-8')

