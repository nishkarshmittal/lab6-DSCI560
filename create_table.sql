-- Active: 1771389284205@@127.0.0.1@5432@lab5_reddit@public
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    reddit_id VARCHAR(50) UNIQUE,
    subreddit VARCHAR(100),
    title TEXT,
    body TEXT,
    image_url TEXT,
    image_path TEXT,
    image_ocr_text TEXT,
    author_masked VARCHAR(128),
    created_utc TIMESTAMP,
    raw_html TEXT,
    cleaned_text TEXT,
    embedding BYTEA,
    cluster_id INT,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
