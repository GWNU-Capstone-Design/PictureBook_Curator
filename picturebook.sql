CREATE TABLE `User` (
    `user_id` INT NOT NULL AUTO_INCREMENT,
    `user_email` VARCHAR(40) NOT NULL,
    `user_name` VARCHAR(10) NOT NULL,
    `user_pw` VARCHAR(20) NOT NULL,
    PRIMARY KEY (`user_id`)
);

CREATE TABLE `Book` (
    `book_id` INT NOT NULL AUTO_INCREMENT,
    `user_id` INT NOT NULL,
    `book_name` VARCHAR(30) NOT NULL,
    PRIMARY KEY (`book_id`),
    FOREIGN KEY (`user_id`) REFERENCES `User` (`user_id`)
);

CREATE TABLE `Image` (
    `image_id` INT NOT NULL AUTO_INCREMENT,
    `book_id` INT NOT NULL,
    `image_value` VARCHAR(255) NOT NULL,
    PRIMARY KEY (`image_id`),
    FOREIGN KEY (`book_id`) REFERENCES `Book` (`book_id`)
);

CREATE TABLE `Text` (
    `text_id` INT NOT NULL,
    `image_id` INT NOT NULL,
    `text_value` JSON,
    PRIMARY KEY (`text_id`),
    FOREIGN KEY (`image_id`) REFERENCES `Image` (`image_id`)
);

CREATE TABLE `Scenario` (
    `scenario_id` INT NOT NULL AUTO_INCREMENT,
    `text_id` INT NOT NULL,
    `scenario_value` TEXT NOT NULL,
    PRIMARY KEY (`scenario_id`),
    FOREIGN KEY (`text_id`) REFERENCES `Text` (`text_id`)
);
