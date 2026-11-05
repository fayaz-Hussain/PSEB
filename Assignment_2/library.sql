-- Create the database and select it
CREATE DATABASE IF NOT EXISTS library;
USE library;

-- Create the books table
CREATE TABLE IF NOT EXISTS books (
    id INT PRIMARY KEY,
    title VARCHAR(100) NOT NULL,
    author VARCHAR(100) NOT NULL,
    year INT,
    price DECIMAL(10, 2)
);

-- Insert 8 book records
INSERT INTO books (id, title, author, year, price) VALUES
(1, 'To Kill a Mockingbird', 'Harper Lee', 1960, 450.00),
(2, '1984', 'George Orwell', 1949, 399.00),
(3, 'The Great Gatsby', 'F. Scott Fitzgerald', 1925, 350.00),
(4, 'Clean Code', 'Robert C. Martin', 2008, 1200.00),
(5, 'The Alchemist', 'Paulo Coelho', 1988, 550.00),
(6, 'Atomic Habits', 'James Clear', 2018, 899.00),
(7, 'Pride and Prejudice', 'Jane Austen', 1813, 300.00),
(8, 'Python Crash Course', 'Eric Matthes', 2015, 1500.00);

-- 1. Display books priced above 500
SELECT * FROM books WHERE price > 500;

-- 2. Display all books sorted by price (highest first)
SELECT * FROM books ORDER BY price DESC;

-- 3. Count the total number of books
SELECT COUNT(*) AS total_books FROM books;

-- 4. Update the price of one book using WHERE
UPDATE books SET price = 499.00 WHERE id = 2;

-- Check the update
SELECT * FROM books WHERE id = 2;