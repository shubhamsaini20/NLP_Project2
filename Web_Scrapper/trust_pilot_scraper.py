import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import concurrent.futures


class TrustPilotScraper:
    

    def __init__(self, url, num_pages):
       
        self.url = url
        self.num_pages = num_pages

    def __extract_url(self, url):
        domain_pattern = r'www\.\S+'
        domain = re.search(domain_pattern, url)
        if domain:
            extracted_domain = domain.group()
            return extracted_domain
        else:
            return url

    def scrape_reviews(self):
       
        base_url = self.url + '?page='
        urls = [base_url + str(i) for i in range(1, self.num_pages + 1)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(self.__scrape_trustpilot_data, urls)

        page_results = list(results)

        reviews, headlines, ratings, authors, dates = [], [], [], [], []
        for page_result in page_results:
            page_reviews, page_headlines, page_ratings, page_authors, page_dates = self.__extract_nested_content(page_result)
            reviews.extend(page_reviews)
            headlines.extend(page_headlines)
            ratings.extend(page_ratings)
            authors.extend(page_authors)
            dates.extend(page_dates)
        self.page_results = page_result

        return reviews, headlines, ratings, authors, dates

    def __scrape_trustpilot_data(self, url):
        
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error scraping URL: {url}\n{str(e)}")
            return None

        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            script_tag = soup.find('script', {'data-business-unit-json-ld': 'true'})
            json_str = script_tag.text.strip()
            json_data = json.loads(json_str)
            return json_data
        except (AttributeError, ValueError, KeyError) as e:
            print(f"Error parsing JSON data for URL: {url}\n{str(e)}")
            return None

    def __get_content_from_page_results(self, page_result, key):
        
        if page_result and '@graph' in page_result:
            page_content = [i[key] for i in page_result['@graph'] if i.get('@type') == 'Review']
            return page_content
        return []

    def __extract_nested_content(self, page_result):
        
        reviews = self.__get_content_from_page_results(page_result, 'reviewBody')
        headlines = self.__get_content_from_page_results(page_result, 'headline')
        ratings = self.__get_content_from_page_results(page_result, 'reviewRating')
        authors = self.__get_content_from_page_results(page_result, 'author')
        dates = self.__get_content_from_page_results(page_result, 'datePublished')
        return reviews, headlines, ratings, authors, dates

    def get_dataframe_results(self, reviews, headlines, ratings, authors, dates):
       
        df = pd.DataFrame({'review': reviews,
                           'author': authors,
                           'headline': headlines,
                           'ratings': ratings,
                           'date': dates})
        df['company'] = self.__extract_url(self.url)
        df_author = self.transform_nested_dataframe_column(df, 'author', ['name', 'url'])
        df_rating = self.transform_nested_dataframe_column(df, 'ratings', ['ratingValue'])
        return pd.concat([df[['company', 'date', 'headline', 'review']], df_author, df_rating], axis=1)


    @staticmethod
    def transform_nested_dataframe_column(df, nested_col, new_cols):
        
        df_flat = df[nested_col].apply(pd.Series)[new_cols]
        return df_flat

    def run(self):
       
        reviews, headlines, ratings, authors, dates = self.scrape_reviews()
        df = self.get_dataframe_results(reviews, headlines, ratings, authors, dates)
        return df