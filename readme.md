# DSCI 560 – Lab 5: Reddit Scraper & Clustering System

For this lab, we built a complete pipeline that scrapes Reddit data, cleans it, stores it in a PostgreSQL database, generates embeddings, and clusters similar posts together. The goal was to collect real forum data and organize it in a way that allows meaningful grouping and analysis.

We chose to scrape Reddit using **old.reddit.com** with BeautifulSoup instead of the official API. The Reddit API has strict limits (1000 posts max and time restrictions), so scraping old Reddit allowed us to fetch larger datasets more reliably while still following polite scraping practices (custom User-Agent, request delays, and retry/backoff handling).

We selected topic-based subreddits (such as r/tech and r/cybersecurity) to keep the domain focused. This makes clustering more meaningful because posts are still related but contain different subtopics.

The pipeline works as follows:

1. Scraping  
   The script takes subreddit names and a number of posts as input.  
   It visits old.reddit.com and follows the “next” button links until the requested number of posts is collected.  
   Promoted posts and advertisements are filtered out.  
   Retry and exponential backoff logic is implemented to handle large requests (5000+ posts) and temporary failures.

2. Preprocessing  
   After scraping, we clean the data:
   - Remove HTML tags  
   - Remove URLs and special characters  
   - Convert text to lowercase  
   - Mask usernames for privacy  
   - Convert timestamps to standard format  

   Some posts contain images (memes). If an image URL is detected, we use Tesseract OCR to extract text from the image and store it in the database as additional content.

3. Database Storage  
   We used PostgreSQL for storage.  
   Each post is saved with:
   - subreddit  
   - title  
   - body  
   - cleaned text  
   - OCR text (if available)  
   - masked author  
   - timestamp  
   - embedding vector (BYTEA format)  
   - cluster ID  
   - extracted keywords  
   - distance to cluster centroid  

   We use `ON CONFLICT DO UPDATE` to ensure that embeddings, clusters, keywords, and distances are updated if a post already exists.  
   An index is created on the centroid distance column for faster querying.

4. Embeddings  
   We used Doc2Vec (Gensim) to convert each cleaned post into a 100-dimensional vector.  
   The trained Doc2Vec model is saved to disk and reused during interactive querying.  
   This transforms raw text into a numerical format that represents semantic meaning.

5. Clustering  
   We implemented KMeans clustering (hard clustering).  
   Each post belongs to exactly one cluster.  
   After clustering:
   - The post closest to each cluster centroid is identified as the representative post.  
   - Top keywords per cluster are extracted using TF-IDF and stored in the database.  

   We also visualize:
   - Cluster distribution using PCA (2D projection of embeddings)  
   - Top keywords per cluster using bar charts  

6. Interactive Query Mode  
   When the script is not running in interval mode, it enters interactive mode.  
   The user can type keywords or a message.  

   The system:
   - Cleans and tokenizes the input  
   - Uses the saved Doc2Vec model to generate an embedding  
   - Finds the closest cluster  
   - Displays representative posts and associated keywords  

   This allows real-time semantic search over clustered Reddit posts.

7. Automation  
   The script supports periodic execution using a command-line argument:

   python old_reddit_scraper.py --interval 5

   This allows the scraper to automatically update the database every X minutes.  
   When not running continuously, it performs a single scrape-and-process cycle and then enables interactive querying.

---

Overall, the system:
- Collects live Reddit data  
- Cleans and preprocesses it  
- Extracts image text  
- Stores everything in PostgreSQL  
- Generates embeddings  
- Extracts keywords  
- Identifies representative posts  
- Clusters similar posts  
- Visualizes cluster structure  
- Supports interactive querying  
- Supports automated updates  

---

This project helped us understand real-world challenges such as API limits, noisy data, advertisements, preprocessing decisions, embedding generation, clustering strategies, visualization, and efficient querying.

To run the project:
1. Install dependencies (see requirements in project folder)  
2. Ensure PostgreSQL is running  
3. Create the required database and table  
4. Run the script with desired subreddits and number of posts  

---

Example:

python old_reddit_scraper.py --subs tech cybersecurity --num 500 --db-host localhost --db-user postgres --db-pass YOUR_PASSWORD --images

That’s the overall workflow of our Lab 5 system.
