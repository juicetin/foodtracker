import { useDetectionStore } from '../useDetectionStore';
import type { DetectedItem } from '../../services/detection/types';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Minimal DetectedItem fixture. */
function makeItem(overrides: Partial<DetectedItem> = {}): DetectedItem {
  return {
    id: 'det_1',
    className: 'Curry',
    confidence: 0.85,
    bbox: { x: 0.1, y: 0.1, w: 0.3, h: 0.3 },
    portionEstimate: {
      weightG: 200,
      confidence: 'high',
      method: 'geometry',
      suggestReference: false,
      details: {},
    },
    portionMultiplier: 1.0,
    isRemoved: false,
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// Reset store between tests
// ---------------------------------------------------------------------------

beforeEach(() => {
  useDetectionStore.getState().reset();
});

// ---------------------------------------------------------------------------
// VLM state tests
// ---------------------------------------------------------------------------

describe('useDetectionStore VLM extensions', () => {
  it('initial isRefining is false', () => {
    expect(useDetectionStore.getState().isRefining).toBe(false);
  });

  it('initial userMealText is empty string', () => {
    expect(useDetectionStore.getState().userMealText).toBe('');
  });

  it('setRefining(true) sets store isRefining to true', () => {
    useDetectionStore.getState().setRefining(true);
    expect(useDetectionStore.getState().isRefining).toBe(true);
  });

  it('setRefining(true) sets isRefining on all active items', () => {
    const items = [makeItem({ id: 'a' }), makeItem({ id: 'b' })];
    useDetectionStore.getState().setItems(items);
    useDetectionStore.getState().setRefining(true);
    const updated = useDetectionStore.getState().items;
    expect(updated[0].isRefining).toBe(true);
    expect(updated[1].isRefining).toBe(true);
  });

  it('setRefining(false) clears item-level isRefining on all items', () => {
    const items = [
      makeItem({ id: 'a', isRefining: true }),
      makeItem({ id: 'b', isRefining: true }),
    ];
    useDetectionStore.getState().setItems(items);
    useDetectionStore.getState().setRefining(false);

    const updated = useDetectionStore.getState().items;
    expect(updated[0].isRefining).toBe(false);
    expect(updated[1].isRefining).toBe(false);
    expect(useDetectionStore.getState().isRefining).toBe(false);
  });

  it('refineItem updates VLM fields on matching item', () => {
    useDetectionStore.getState().setItems([makeItem({ id: 'det_1', isRefining: true })]);

    useDetectionStore.getState().refineItem('det_1', {
      vlmLabel: 'Massaman Curry',
      vlmCuisine: 'Thai',
      vlmIngredients: ['peanut', 'potato', 'coconut milk'],
      vlmConfidence: 0.92,
    });

    const item = useDetectionStore.getState().items[0];
    expect(item.vlmLabel).toBe('Massaman Curry');
    expect(item.vlmCuisine).toBe('Thai');
    expect(item.vlmIngredients).toEqual(['peanut', 'potato', 'coconut milk']);
    expect(item.vlmConfidence).toBe(0.92);
  });

  it('refineItem sets item isRefining to false', () => {
    useDetectionStore
      .getState()
      .setItems([makeItem({ id: 'det_1', isRefining: true })]);

    useDetectionStore.getState().refineItem('det_1', {
      vlmLabel: 'Massaman Curry',
    });

    expect(useDetectionStore.getState().items[0].isRefining).toBe(false);
  });

  it('refineItem leaves other items unchanged', () => {
    const items = [
      makeItem({ id: 'a', className: 'Rice' }),
      makeItem({ id: 'b', className: 'Curry', isRefining: true }),
    ];
    useDetectionStore.getState().setItems(items);

    useDetectionStore.getState().refineItem('b', {
      vlmLabel: 'Green Curry',
      vlmCuisine: 'Thai',
    });

    const result = useDetectionStore.getState().items;
    // Item 'a' should be untouched
    expect(result[0].className).toBe('Rice');
    expect(result[0].vlmLabel).toBeUndefined();
    // Item 'b' should be refined
    expect(result[1].vlmLabel).toBe('Green Curry');
  });

  it('setUserText updates userMealText', () => {
    useDetectionStore.getState().setUserText('big bowl of laksa');
    expect(useDetectionStore.getState().userMealText).toBe('big bowl of laksa');
  });

  it('displayLabel returns vlmLabel when present', () => {
    const item = makeItem({ vlmLabel: 'Massaman Curry', className: 'Curry' });
    expect(useDetectionStore.getState().displayLabel(item)).toBe(
      'Massaman Curry',
    );
  });

  it('displayLabel returns empty string when isRefining (shimmer state)', () => {
    const item = makeItem({ isRefining: true });
    expect(useDetectionStore.getState().displayLabel(item)).toBe('');
  });

  it('displayLabel returns "Unknown food" when no vlmLabel and not refining', () => {
    const item = makeItem({ className: 'Food Region' });
    expect(useDetectionStore.getState().displayLabel(item)).toBe('Unknown food');
  });

  it('reset clears VLM state', () => {
    useDetectionStore.getState().setRefining(true);
    useDetectionStore.getState().setUserText('something');
    useDetectionStore.getState().reset();

    expect(useDetectionStore.getState().isRefining).toBe(false);
    expect(useDetectionStore.getState().userMealText).toBe('');
  });

  it('existing removeItem still works (backward compat)', () => {
    useDetectionStore.getState().setItems([makeItem({ id: 'det_1' })]);
    useDetectionStore.getState().removeItem('det_1');

    const item = useDetectionStore.getState().items[0];
    expect(item.isRemoved).toBe(true);
    expect(item.removedAt).toBeDefined();
  });
});
