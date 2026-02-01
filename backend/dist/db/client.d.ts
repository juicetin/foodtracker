import pg from 'pg';
declare const pool: import("pg").Pool;
export default pool;
export declare function query(text: string, params?: any[]): Promise<import("pg").QueryResult<any>>;
export declare function transaction<T>(callback: (client: pg.PoolClient) => Promise<T>): Promise<T>;
