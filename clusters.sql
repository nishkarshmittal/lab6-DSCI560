-- Active: 1771389415215@@127.0.0.1@5432@lab5_reddit@public
SELECT cluster_id, COUNT(*)
FROM posts
GROUP BY cluster_id
ORDER BY cluster_id;
