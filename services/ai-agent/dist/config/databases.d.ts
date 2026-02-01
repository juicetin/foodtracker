export interface FoodDatabase {
    id: string;
    name: string;
    region: string;
    apiEndpoint?: string;
    priority: number;
}
export declare const FOOD_DATABASES: FoodDatabase[];
export declare function getDatabaseForRegion(region: string): FoodDatabase[];
export declare const FOOD_DENSITY_TABLE: Record<string, number>;
