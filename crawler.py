import requests
from bs4 import BeautifulSoup
import json
import time

def crawl_wikipedia(start_url, max_pages=20, output_file='output.json'):
    visited_pages = [] # list of the visited URLs
    pages_to_visit = [start_url] # list of URLs to visit

    while pages_to_visit and len(visited_pages) < max_pages:
        current_url = pages_to_visit.pop(0)

        try:
            response = requests.get(current_url)
            if response.status_code != 200:
                print(f"Σφάλμα HTTP: {response.status_code} στη σελίδα {current_url}")
                continue

            soup = BeautifulSoup(response.content, 'html.parser') # html analy

            # Titles
            title = soup.find('h1').get_text()
            visited_pages.append({"url": current_url, "title": title})

            # Links
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/wiki/') and ':' not in href:
                    full_url = f"https://el.wikipedia.org{href}"
                    if full_url not in [page['url'] for page in visited_pages] and full_url not in pages_to_visit:
                        pages_to_visit.append(full_url)
            
            # Wait for 1 second before next request
            time.sleep(1)

        except Exception as e:
            print(f"Σφάλμα κατα την Επεξεργασία της σελίδας {current_url}: {e}")


            # Save progress to JSON file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(visited_pages, file, ensure_ascii=False, indent=4)

    print (f"Τα δεδομένα αποθηκεύτηκαν στο {output_file}")

if __name__ == "__main__":
    start_url = 'https://el.wikipedia.org/wiki/Python'
    crawl_wikipedia(start_url)