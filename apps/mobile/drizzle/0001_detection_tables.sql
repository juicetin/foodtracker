CREATE TABLE IF NOT EXISTS correction_history (
  id TEXT PRIMARY KEY,
  original_class_name TEXT NOT NULL,
  corrected_class_name TEXT NOT NULL,
  confidence REAL NOT NULL,
  corrected_at TEXT DEFAULT (datetime('now'))
);
