import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

# URL of the main page to scrape
main_url = 'https://phys.org/biology-news/'

# Define headers to mimic a browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Send a GET request to the main page with headers
response = requests.get(main_url, headers=headers)

# Check the response status code
if response.status_code != 200:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")
else:
    # Parse the HTML content of the main page with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all h3 elements with the specified class
    h3_elements = soup.find_all('h3', class_='mb-1 mb-lg-2')

    if not h3_elements:
        print("No <h3> elements found with the specified class.")
    else:
        # Extract the href attribute from the first a tag within the first h3 element
        hrefs = [h3.find('a')['href'] for h3 in h3_elements if h3.find('a')]

        # Check if there are any hrefs
        if not hrefs:
            print("No hrefs found within the <h3> elements.")
        else:
            first_href = hrefs[0]  # Get the first href
            full_url = requests.compat.urljoin(main_url, first_href)  # Construct the full URL

            # Print the full URL for debugging
            print(f"Fetching content from: {full_url}")

            # Send a GET request to the linked page
            linked_response = requests.get(full_url, headers=headers)

            # Check the response status code
            if linked_response.status_code != 200:
                print(f"Failed to retrieve the linked page. Status code: {linked_response.status_code}")
            else:
                # Parse the HTML content of the linked page with BeautifulSoup
                linked_soup = BeautifulSoup(linked_response.content, 'html.parser')

                # Extract metadata
                title = linked_soup.find('title').text if linked_soup.find('title') else 'No title'
                description = linked_soup.find('meta', attrs={'name': 'description'})['content'] if linked_soup.find('meta', attrs={'name': 'description'}) else 'No description'
                source = full_url
                language = linked_soup.html.get('lang', 'en')  # Default to 'en' if no language is specified

                # Find the <div> with the class 'mt-4 article-main'
                gallery_div = linked_soup.find('div', class_='mt-4 article-main')

                if gallery_div is None:
                    print("No <div> with the class 'mt-4 article-main' found on the linked page.")
                else:
                    # Exclude specific nested <div> elements
                    excluded_div_classes = ['article-gallery', 'article-banner']
                    text_content = []
                    for p in gallery_div.find_all('p', recursive=False):
                        parent_div = p.find_parent('div')
                        if not parent_div or not any(cls in parent_div.get('class', []) for cls in excluded_div_classes):
                            text_content.append(p.get_text())

                    if not text_content:
                        print("No text content found.")
                    else:
                        # Combine all paragraphs into one document
                        document = "\n".join(text_content)
                        
                        # Create the text splitter
                        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                            chunk_size=250, chunk_overlap=0
                        )
                        
                        # Split the document into chunks
                        doc_splits = text_splitter.split_text(document)
                        
                        # Format chunks with metadata
                        formatted_chunks = []
                        for idx, split in enumerate(doc_splits):
                            formatted_chunk = {
                                'page_content': split,
                                'metadata': {
                                    'source': source,
                                    'title': title,
                                    'description': description,
                                    'language': language
                                }
                            }
                            formatted_chunks.append(formatted_chunk)
                        
                        # Print formatted chunks for inspection
                        for i, chunk in enumerate(formatted_chunks):
                            print(f"Chunk {i}:")
                            print(f"page_content='{chunk['page_content']}' metadata={chunk['metadata']}\n")
                        for i, split in enumerate(doc_splits):
                                     print(f"Chunk {i}:\n{split}\n")     
                        
