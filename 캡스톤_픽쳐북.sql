CREATE DATABASE PICTUREBOOK;
USE PICTUREBOOK;

CREATE TABLE `User` (
    `user_id` INT NOT NULL AUTO_INCREMENT,
    `user_email` VARCHAR(40) NOT NULL,
    `user_name` VARCHAR(10) NOT NULL,
    `user_pw` VARCHAR(20) NOT NULL,
    PRIMARY KEY (`user_id`)
);

CREATE TABLE `Book` (
    `book_id` INT NOT NULL AUTO_INCREMENT,
    `book_name` VARCHAR(30) NOT NULL,
    `user_id` INT NOT NULL,
    PRIMARY KEY (`book_id`),
    FOREIGN KEY (`user_id`) REFERENCES `User` (`user_id`)
);

CREATE TABLE `Image` (
    `image_id` INT NOT NULL AUTO_INCREMENT,
    `book_id` INT NOT NULL,
    `image_value_1` VARCHAR(255) NOT NULL,
    `image_value_2` VARCHAR(255) NOT NULL,
    `image_value_3` VARCHAR(255) NOT NULL,
    `image_value_4` VARCHAR(255) NOT NULL,
    `image_value_5` VARCHAR(255) NOT NULL,
    `image_value_6` VARCHAR(255) NOT NULL,
    `image_value_7` VARCHAR(255) NOT NULL,
    `image_value_8` VARCHAR(255) NOT NULL,
    `image_value_9` VARCHAR(255) NOT NULL,
    PRIMARY KEY (`image_id`),
    FOREIGN KEY (`book_id`) REFERENCES `Book` (`book_id`)
);

CREATE TABLE `Scenario` (
    `scenario_id` INT NOT NULL AUTO_INCREMENT,
    `image_id` INT NOT NULL,
    `scenario_value` TEXT NOT NULL,
    PRIMARY KEY (`scenario_id`),
    FOREIGN KEY (`image_id`) REFERENCES `Image` (`image_id`)
);
