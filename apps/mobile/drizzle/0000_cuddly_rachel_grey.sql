CREATE TABLE `container_weights` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`name` text NOT NULL,
	`weight_grams` real NOT NULL,
	`times_used` integer DEFAULT 0,
	`last_used_at` text,
	`created_at` text DEFAULT (datetime('now'))
);
--> statement-breakpoint
CREATE TABLE `custom_recipes` (
	`id` text PRIMARY KEY NOT NULL,
	`name` text NOT NULL,
	`description` text,
	`source_entry_id` text,
	`total_calories` real DEFAULT 0,
	`total_protein` real DEFAULT 0,
	`total_carbs` real DEFAULT 0,
	`total_fat` real DEFAULT 0,
	`times_used` integer DEFAULT 0,
	`last_used_at` text,
	`created_at` text DEFAULT (datetime('now')),
	`updated_at` text DEFAULT (datetime('now')),
	FOREIGN KEY (`source_entry_id`) REFERENCES `food_entries`(`id`) ON UPDATE no action ON DELETE set null
);
--> statement-breakpoint
CREATE TABLE `food_entries` (
	`id` text PRIMARY KEY NOT NULL,
	`meal_type` text NOT NULL,
	`entry_date` text NOT NULL,
	`total_calories` real DEFAULT 0,
	`total_protein` real DEFAULT 0,
	`total_carbs` real DEFAULT 0,
	`total_fat` real DEFAULT 0,
	`notes` text,
	`updated_at` text DEFAULT (datetime('now')),
	`is_synced` integer DEFAULT false,
	`is_deleted` integer DEFAULT false,
	`created_at` text DEFAULT (datetime('now'))
);
--> statement-breakpoint
CREATE TABLE `ingredients` (
	`id` text PRIMARY KEY NOT NULL,
	`entry_id` text NOT NULL,
	`name` text NOT NULL,
	`quantity` real NOT NULL,
	`unit` text NOT NULL,
	`calories` real NOT NULL,
	`protein` real DEFAULT 0,
	`carbs` real DEFAULT 0,
	`fat` real DEFAULT 0,
	`fiber` real DEFAULT 0,
	`sugar` real DEFAULT 0,
	`ai_confidence` real,
	`bounding_box_x` real,
	`bounding_box_y` real,
	`bounding_box_w` real,
	`bounding_box_h` real,
	`database_source` text,
	`database_id` text,
	`user_modified` integer DEFAULT false,
	`original_quantity` real,
	`updated_at` text DEFAULT (datetime('now')),
	`created_at` text DEFAULT (datetime('now')),
	FOREIGN KEY (`entry_id`) REFERENCES `food_entries`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE TABLE `installed_packs` (
	`id` text PRIMARY KEY NOT NULL,
	`name` text NOT NULL,
	`type` text NOT NULL,
	`version` text NOT NULL,
	`file_path` text NOT NULL,
	`size_bytes` integer,
	`sha256` text,
	`region` text,
	`installed_at` text DEFAULT (datetime('now')),
	`last_checked` text
);
--> statement-breakpoint
CREATE TABLE `model_cache` (
	`model_id` text PRIMARY KEY NOT NULL,
	`version` text NOT NULL,
	`file_path` text NOT NULL,
	`size_bytes` integer,
	`downloaded_at` text DEFAULT (datetime('now'))
);
--> statement-breakpoint
CREATE TABLE `modification_history` (
	`id` text PRIMARY KEY NOT NULL,
	`ingredient_id` text NOT NULL,
	`field_name` text NOT NULL,
	`old_value` text,
	`new_value` text,
	`modified_at` text DEFAULT (datetime('now')),
	FOREIGN KEY (`ingredient_id`) REFERENCES `ingredients`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE TABLE `photo_hashes` (
	`photo_id` text PRIMARY KEY NOT NULL,
	`phash` text NOT NULL,
	`created_at` text DEFAULT (datetime('now')),
	FOREIGN KEY (`photo_id`) REFERENCES `photos`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE TABLE `photos` (
	`id` text PRIMARY KEY NOT NULL,
	`entry_id` text NOT NULL,
	`uri` text NOT NULL,
	`local_path` text,
	`width` integer,
	`height` integer,
	`latitude` real,
	`longitude` real,
	`uploaded_at` text DEFAULT (datetime('now')),
	FOREIGN KEY (`entry_id`) REFERENCES `food_entries`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE TABLE `recipe_ingredients` (
	`id` text PRIMARY KEY NOT NULL,
	`recipe_id` text NOT NULL,
	`name` text NOT NULL,
	`quantity` real NOT NULL,
	`unit` text NOT NULL,
	`calories` real NOT NULL,
	`protein` real DEFAULT 0,
	`carbs` real DEFAULT 0,
	`fat` real DEFAULT 0,
	`created_at` text DEFAULT (datetime('now')),
	FOREIGN KEY (`recipe_id`) REFERENCES `custom_recipes`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE TABLE `recipe_photos` (
	`id` text PRIMARY KEY NOT NULL,
	`recipe_id` text NOT NULL,
	`local_path` text NOT NULL,
	`is_primary` integer DEFAULT false,
	`created_at` text DEFAULT (datetime('now')),
	FOREIGN KEY (`recipe_id`) REFERENCES `custom_recipes`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE TABLE `scan_queue` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`asset_id` text,
	`uri` text NOT NULL,
	`status` text DEFAULT 'pending' NOT NULL,
	`created_at` text DEFAULT (datetime('now')),
	`processed_at` text
);
--> statement-breakpoint
CREATE TABLE `sync_outbox` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`table_name` text NOT NULL,
	`record_id` text NOT NULL,
	`operation` text NOT NULL,
	`created_at` text DEFAULT (datetime('now'))
);
--> statement-breakpoint
CREATE TABLE `user_settings` (
	`id` integer PRIMARY KEY AUTOINCREMENT NOT NULL,
	`region` text DEFAULT 'AU',
	`units` text DEFAULT 'metric',
	`calorie_goal` real DEFAULT 2000,
	`protein_goal` real DEFAULT 150,
	`carbs_goal` real DEFAULT 200,
	`fat_goal` real DEFAULT 65,
	`created_at` text DEFAULT (datetime('now')),
	`updated_at` text DEFAULT (datetime('now'))
);
