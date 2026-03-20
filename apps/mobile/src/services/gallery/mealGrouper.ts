/**
 * Meal grouper: clusters classified food photos into meal events
 * based on temporal proximity (1-hour window) and GPS proximity (150m).
 *
 * When either photo in a pair lacks GPS, grouping falls back to time-only.
 */

import type { ClassifiedPhoto, GalleryScanPreferences, MealGroup } from './types';
import { DEFAULT_SCAN_PREFS } from './types';

// ---------------------------------------------------------------------------
// Haversine distance (meters)
// ---------------------------------------------------------------------------

const EARTH_RADIUS_M = 6_371_000;

/**
 * Compute great-circle distance between two lat/lon points in meters.
 */
export function haversineDistance(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number,
): number {
  const toRad = (deg: number) => (deg * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return EARTH_RADIUS_M * c;
}

// ---------------------------------------------------------------------------
// Meal grouping
// ---------------------------------------------------------------------------

function hasGps(photo: ClassifiedPhoto): boolean {
  return photo.latitude != null && photo.longitude != null;
}

/**
 * Group classified food photos into meal events.
 *
 * Algorithm: sort by creationTime, then for each photo try to merge into an
 * existing group. A photo merges if:
 *   (a) time diff to ANY photo in the group <= mealWindowMs, AND
 *   (b) if both have GPS, distance <= gpsProximityM; if either lacks GPS, merge by time only.
 */
export function groupIntoMeals(
  photos: ClassifiedPhoto[],
  prefs?: Partial<GalleryScanPreferences>,
): MealGroup[] {
  if (photos.length === 0) return [];

  const { mealWindowMs, gpsProximityM } = { ...DEFAULT_SCAN_PREFS, ...prefs };

  // Sort ascending by creationTime
  const sorted = [...photos].sort((a, b) => a.creationTime - b.creationTime);

  const groups: MealGroup[] = [];

  for (const photo of sorted) {
    let merged = false;

    for (const group of groups) {
      // Check time proximity against any photo in the group
      const withinTime = group.photos.some(
        (gp) => Math.abs(photo.creationTime - gp.creationTime) <= mealWindowMs,
      );
      if (!withinTime) continue;

      // Check GPS proximity (if both have GPS)
      const bothHaveGps = hasGps(photo) && group.photos.some(hasGps);
      if (bothHaveGps) {
        const gpsPhotos = group.photos.filter(hasGps);
        const withinGps = gpsPhotos.some(
          (gp) =>
            haversineDistance(
              photo.latitude!,
              photo.longitude!,
              gp.latitude!,
              gp.longitude!,
            ) <= gpsProximityM,
        );
        if (!withinGps) continue;
      }

      // Merge into this group
      group.photos.push(photo);
      group.lastTimestamp = Math.max(group.lastTimestamp, photo.creationTime);
      group.firstTimestamp = Math.min(group.firstTimestamp, photo.creationTime);

      // Update location from first photo with GPS
      if (!group.location && hasGps(photo)) {
        group.location = { latitude: photo.latitude!, longitude: photo.longitude! };
      }

      merged = true;
      break;
    }

    if (!merged) {
      groups.push({
        id: crypto.randomUUID(),
        photos: [photo],
        firstTimestamp: photo.creationTime,
        lastTimestamp: photo.creationTime,
        location: hasGps(photo)
          ? { latitude: photo.latitude!, longitude: photo.longitude! }
          : undefined,
      });
    }
  }

  return groups;
}
