import { groupIntoMeals, haversineDistance } from '../mealGrouper';
import type { ClassifiedPhoto } from '../types';

// Helper: create a classified food photo at a given time/location
function makePhoto(
  overrides: Partial<ClassifiedPhoto> & { creationTime: number },
): ClassifiedPhoto {
  return {
    id: Math.floor(Math.random() * 100000),
    assetId: `asset-${Math.random().toString(36).slice(2, 8)}`,
    uri: 'file:///photo.jpg',
    isFood: true,
    latitude: null,
    longitude: null,
    ...overrides,
  };
}

const THIRTY_MIN = 30 * 60 * 1000;
const TWO_HOURS = 2 * 60 * 60 * 1000;
const BASE_TIME = Date.now();

describe('haversineDistance', () => {
  it('returns expected distance for known coordinates within 1% tolerance', () => {
    // Sydney Opera House (-33.8568, 151.2153) to Sydney Tower Eye (-33.8708, 151.2089) ~ 1.66 km
    const dist = haversineDistance(-33.8568, 151.2153, -33.8708, 151.2089);
    expect(dist).toBeGreaterThan(1640);
    expect(dist).toBeLessThan(1690);
  });

  it('returns 0 for identical coordinates', () => {
    expect(haversineDistance(0, 0, 0, 0)).toBe(0);
  });
});

describe('groupIntoMeals', () => {
  it('groups 2 photos 30min apart with same GPS into 1 meal', () => {
    const photos = [
      makePhoto({ creationTime: BASE_TIME, latitude: -33.85, longitude: 151.21 }),
      makePhoto({ creationTime: BASE_TIME + THIRTY_MIN, latitude: -33.85, longitude: 151.21 }),
    ];
    const groups = groupIntoMeals(photos);
    expect(groups).toHaveLength(1);
    expect(groups[0].photos).toHaveLength(2);
  });

  it('separates 2 photos 2hr apart into different meals', () => {
    const photos = [
      makePhoto({ creationTime: BASE_TIME }),
      makePhoto({ creationTime: BASE_TIME + TWO_HOURS }),
    ];
    const groups = groupIntoMeals(photos);
    expect(groups).toHaveLength(2);
  });

  it('groups 2 photos 30min apart when one is missing GPS (time-only fallback)', () => {
    const photos = [
      makePhoto({ creationTime: BASE_TIME, latitude: -33.85, longitude: 151.21 }),
      makePhoto({ creationTime: BASE_TIME + THIRTY_MIN, latitude: null, longitude: null }),
    ];
    const groups = groupIntoMeals(photos);
    expect(groups).toHaveLength(1);
    expect(groups[0].photos).toHaveLength(2);
  });

  it('separates 2 photos 30min apart when GPS is 500m apart', () => {
    // ~500m apart
    const photos = [
      makePhoto({ creationTime: BASE_TIME, latitude: -33.8500, longitude: 151.2100 }),
      makePhoto({ creationTime: BASE_TIME + THIRTY_MIN, latitude: -33.8545, longitude: 151.2100 }),
    ];
    const groups = groupIntoMeals(photos);
    expect(groups).toHaveLength(2);
  });

  it('groups 5 photos spanning 45min into 1 meal', () => {
    const photos = Array.from({ length: 5 }, (_, i) =>
      makePhoto({ creationTime: BASE_TIME + i * 11 * 60 * 1000 }), // ~11min apart
    );
    const groups = groupIntoMeals(photos);
    expect(groups).toHaveLength(1);
    expect(groups[0].photos).toHaveLength(5);
  });

  it('returns empty array for empty input', () => {
    expect(groupIntoMeals([])).toEqual([]);
  });

  it('sets firstTimestamp and lastTimestamp correctly', () => {
    const photos = [
      makePhoto({ creationTime: BASE_TIME }),
      makePhoto({ creationTime: BASE_TIME + THIRTY_MIN }),
    ];
    const groups = groupIntoMeals(photos);
    expect(groups[0].firstTimestamp).toBe(BASE_TIME);
    expect(groups[0].lastTimestamp).toBe(BASE_TIME + THIRTY_MIN);
  });

  it('sets location from photos with GPS', () => {
    const photos = [
      makePhoto({ creationTime: BASE_TIME, latitude: -33.85, longitude: 151.21 }),
      makePhoto({ creationTime: BASE_TIME + THIRTY_MIN, latitude: -33.85, longitude: 151.21 }),
    ];
    const groups = groupIntoMeals(photos);
    expect(groups[0].location).toEqual({ latitude: -33.85, longitude: 151.21 });
  });
});
