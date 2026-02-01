import scrapy

class NFLTeamStandingsSpider(scrapy.Spider):
  name = 'nfl_team_standings'
  start_urls = ['https://www.pro-football-reference.com/years/2025/']
  def parse(self, response):
    division = ''
    for row in response.css('#AFC tbody tr'):
      item = {}
      if (row.css('td[data-stat=onecell]::text').get() is not None):
        division = row.css('td[data-stat=onecell]::text').get()
      elif row.css('th[data-stat=team] a::text').get() is None:
        continue
      else:
        item['division'] = division
        item['team'] = row.css('th[data-stat=team] a::text').get()
        item['wins'] = row.css('td[data-stat=wins]::text').get()
        item['losses'] = row.css('td[data-stat=losses]::text').get()
        item['ties'] = row.css('td[data-stat=ties]::text').get()
        item['win_loss_percentage'] = row.css('td[data-stat=win_loss_perc]::text').get()
        item['points_scored_by_team'] = row.css('td[data-stat=points]::text').get()
        item['points_scored_by_opposition'] = row.css('td[data-stat=points_opp]::text').get()
        item['points_differential'] = row.css('td[data-stat=points_diff]::text').get()
        item['margin_of_victory'] = row.css('td[data-stat=mov]::text').get()
        item['strength_of_schedule'] = row.css('td[data-stat=sos_total]::text').get()
        item['simple_rating_system'] = row.css('td[data-stat=srs_total]::text').get()
        item['simple_rating_system_offense'] = row.css('td[data-stat=srs_offense]::text').get()
        item['simple_rating_system_defense'] = row.css('td[data-stat=srs_defense]::text').get()

        yield item