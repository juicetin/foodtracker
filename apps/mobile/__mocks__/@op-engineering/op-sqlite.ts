/**
 * Mock for @op-engineering/op-sqlite
 * Simulates open(), execute(), executeAsync() with in-memory state for unit tests.
 */

interface MockResult {
  rows: Record<string, unknown>[];
  rowsAffected: number;
  insertId?: number;
}

interface MockDB {
  execute: jest.Mock<MockResult, [string, unknown[]?]>;
  executeAsync: jest.Mock<Promise<MockResult>, [string, unknown[]?]>;
  close: jest.Mock<void, []>;
  _name: string;
}

const databases = new Map<string, MockDB>();

function createMockDb(name: string): MockDB {
  const db: MockDB = {
    _name: name,
    execute: jest.fn((_sql: string, _params?: unknown[]): MockResult => {
      return { rows: [], rowsAffected: 0 };
    }),
    executeAsync: jest.fn(
      async (_sql: string, _params?: unknown[]): Promise<MockResult> => {
        return { rows: [], rowsAffected: 0 };
      }
    ),
    close: jest.fn(),
  };
  return db;
}

export function open(options: { name: string }): MockDB {
  const existing = databases.get(options.name);
  if (existing) return existing;
  const db = createMockDb(options.name);
  databases.set(options.name, db);
  return db;
}

// Utility for tests to clear state between runs
export function __resetMockDatabases(): void {
  databases.clear();
}

export default { open, __resetMockDatabases };
