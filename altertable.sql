-- Active: 1771389284205@@127.0.0.1@5432@lab5_reddit@public
ALTER TABLE posts ADD COLUMN keywords TEXT;
ALTER TABLE posts ADD COLUMN distance_to_centroid FLOAT;
