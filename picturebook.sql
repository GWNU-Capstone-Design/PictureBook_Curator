CREATE TABLE `User` (
	`user_id`	INT(100)	NOT NULL,
	`user_email`	VARCHAR(40)	NOT NULL,
	`user_name`	VARCHAR(10)	NOT NULL,
	`user_pw`	VARCHAR(20)	NOT NULL
);

CREATE TABLE `Book` (
	`book_id`	INT(100)	NOT NULL,
	`user_id`	INT(100)	NOT NULL,
	`book_name`	VARCHAR(30)	NOT NULL
);

CREATE TABLE `Text` (
	`text_id`	INT(100)	NOT NULL,
	`image_id`	INT(100)	NOT NULL,
	`book_id`	INT(100)	NOT NULL,
	`user_id`	INT(100)	NOT NULL,
	`text_value`	JSON	NOT NULL
);

CREATE TABLE `Image` (
	`image_id`	INT(100)	NOT NULL,
	`book_id`	INT(100)	NOT NULL,
	`user_id`	INT(100)	NOT NULL,
	`image_value`	VARCHAR(255)	NOT NULL'
);

CREATE TABLE `Scenario` (
	`scenario_id`	INT(100)	NOT NULL,
	`text_id`	INT(100)	NOT NULL,
	`image_id`	INT(100)	NOT NULL,
	`book_id`	INT(100)	NOT NULL,
	`user_id`	INT(100)	NOT NULL,
	`scenario_value`	TEXT	NOT NULL
);

ALTER TABLE `User` ADD CONSTRAINT `PK_USER` PRIMARY KEY (
	`user_id`
);

ALTER TABLE `Book` ADD CONSTRAINT `PK_BOOK` PRIMARY KEY (
	`book_id`,
	`user_id`
);

ALTER TABLE `Text` ADD CONSTRAINT `PK_TEXT` PRIMARY KEY (
	`text_id`,
	`image_id`,
	`book_id`,
	`user_id`
);

ALTER TABLE `Image` ADD CONSTRAINT `PK_IMAGE` PRIMARY KEY (
	`image_id`,
	`book_id`,
	`user_id`
);

ALTER TABLE `Scenario` ADD CONSTRAINT `PK_SCENARIO` PRIMARY KEY (
	`scenario_id`,
	`text_id`,
	`image_id`,
	`book_id`,
	`user_id`
);

ALTER TABLE `Book` ADD CONSTRAINT `FK_User_TO_Book_1` FOREIGN KEY (
	`user_id`
)
REFERENCES `User` (
	`user_id`
);

ALTER TABLE `Text` ADD CONSTRAINT `FK_Image_TO_Text_1` FOREIGN KEY (
	`image_id`
)
REFERENCES `Image` (
	`image_id`
);

ALTER TABLE `Text` ADD CONSTRAINT `FK_Image_TO_Text_2` FOREIGN KEY (
	`book_id`
)
REFERENCES `Image` (
	`book_id`
);

ALTER TABLE `Text` ADD CONSTRAINT `FK_Image_TO_Text_3` FOREIGN KEY (
	`user_id`
)
REFERENCES `Image` (
	`user_id`
);

ALTER TABLE `Image` ADD CONSTRAINT `FK_Book_TO_Image_1` FOREIGN KEY (
	`book_id`
)
REFERENCES `Book` (
	`book_id`
);

ALTER TABLE `Image` ADD CONSTRAINT `FK_Book_TO_Image_2` FOREIGN KEY (
	`user_id`
)
REFERENCES `Book` (
	`user_id`
);

ALTER TABLE `Scenario` ADD CONSTRAINT `FK_Text_TO_Scenario_1` FOREIGN KEY (
	`text_id`
)
REFERENCES `Text` (
	`text_id`
);

ALTER TABLE `Scenario` ADD CONSTRAINT `FK_Text_TO_Scenario_2` FOREIGN KEY (
	`image_id`
)
REFERENCES `Text` (
	`image_id`
);

ALTER TABLE `Scenario` ADD CONSTRAINT `FK_Text_TO_Scenario_3` FOREIGN KEY (
	`book_id`
)
REFERENCES `Text` (
	`book_id`
);

ALTER TABLE `Scenario` ADD CONSTRAINT `FK_Text_TO_Scenario_4` FOREIGN KEY (
	`user_id`
)
REFERENCES `Text` (
	`user_id`
);
