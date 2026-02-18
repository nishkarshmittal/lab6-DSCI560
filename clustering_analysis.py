#!/usr/bin/env python3

import psycopg2
import argparse


def get_db_conn(host, user, password, database="lab5_reddit"):
    conn = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password
    )
    conn.autocommit = True
    return conn


def run_query(cursor, title, query):
    print(f"\n{title}")
    cursor.execute(query)
    rows = cursor.fetchall()
    for row in rows:
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-host", required=True)
    parser.add_argument("--db-user", required=True)
    parser.add_argument("--db-pass", required=True)
    args = parser.parse_args()

    conn = get_db_conn(args.db_host, args.db_user, args.db_pass)
    cursor = conn.cursor()

    #basic query
    run_query(
        cursor,
        "No. of entires in posts table",
        """
        SELECT COUNT(*) FROM posts;
        """
    )

    # 1. Cluster distribution with percentages
    run_query(
        cursor,
        "Cluster Distribution",
        """
        SELECT 
            cluster_id,
            COUNT(*) AS total_posts,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 2) AS percentage
        FROM posts
        GROUP BY cluster_id
        ORDER BY total_posts DESC;
        """
    )

    # 2. Representative post per cluster
    run_query(
        cursor,
        "Representative Post Per Cluster",
        """
        SELECT DISTINCT ON (cluster_id)
            cluster_id,
            title,
            distance_to_centroid
        FROM posts
        ORDER BY cluster_id, distance_to_centroid ASC;
        """
    )

    # 3. Top 5 most central posts overall
    run_query(
        cursor,
        "Top 5 Most Central Posts",
        """
        SELECT 
            title,
            cluster_id,
            distance_to_centroid
        FROM posts
        ORDER BY distance_to_centroid ASC
        LIMIT 5;
        """
    )

    # 4. Top 5 Outlier Posts
    run_query(
        cursor,
        "Top 5 Outlier Posts",
        """
        SELECT 
            title,
            cluster_id,
            distance_to_centroid
        FROM posts
        ORDER BY distance_to_centroid DESC
        LIMIT 5;
        """
    )

    # 5. Keywords per cluster
    run_query(
        cursor,
        "Keywords Per Cluster",
        """
        SELECT 
            cluster_id,
            keywords,
            COUNT(*) AS posts_in_cluster
        FROM posts
        GROUP BY cluster_id, keywords
        ORDER BY cluster_id;
        """
    )

    # 6. Search example (AI keyword)
    run_query(
        cursor,
        "Posts Matching Keyword 'ai'",
        """
        SELECT 
            title,
            cluster_id,
            keywords
        FROM posts
        WHERE keywords ILIKE '%ai%'
        ORDER BY cluster_id;
        """
    )

    # 7. Subreddit distribution across clusters
    run_query(
        cursor,
        "Subreddit Distribution Across Clusters",
        """
        SELECT 
            subreddit,
            cluster_id,
            COUNT(*) AS count
        FROM posts
        GROUP BY subreddit, cluster_id
        ORDER BY subreddit, count DESC;
        """
    )

    # 8. Average distance per cluster
    run_query(
        cursor,
        "Cluster Tightness (Average Distance)",
        """
        SELECT 
            cluster_id,
            ROUND(AVG(distance_to_centroid)::numeric, 4) AS avg_distance
        FROM posts
        GROUP BY cluster_id
        ORDER BY avg_distance;
        """
    )

    # 9. Most common keyword groups
    run_query(
        cursor,
        "Most Common Keyword Groups",
        """
        SELECT 
            keywords,
            COUNT(*) AS frequency
        FROM posts
        GROUP BY keywords
        ORDER BY frequency DESC
        LIMIT 10;
        """
    )

    # 10. Cluster coherence stats
    run_query(
        cursor,
        "Cluster Coherence Statistics",
        """
        SELECT 
            cluster_id,
            MIN(distance_to_centroid) AS closest,
            MAX(distance_to_centroid) AS farthest,
            AVG(distance_to_centroid) AS average
        FROM posts
        GROUP BY cluster_id
        ORDER BY cluster_id;
        """
    )

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()
