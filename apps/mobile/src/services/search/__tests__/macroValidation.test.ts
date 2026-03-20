import { validateMacros } from '../historyService';

describe('validateMacros', () => {
  it('returns valid when macros approximately match calories', () => {
    // 25P*4 + 25C*4 + 20F*9 = 100 + 100 + 180 = 380
    const result = validateMacros(400, 25, 25, 20);
    expect(result.isValid).toBe(true);
    expect(result.expected).toBe(380);
  });

  it('flags mismatch when calories far from expected', () => {
    // 10P*4 + 10C*4 + 10F*9 = 40 + 40 + 90 = 170
    const result = validateMacros(1000, 10, 10, 10);
    expect(result.isValid).toBe(false);
    expect(result.expected).toBe(170);
  });

  it('returns valid when all fields are 0 (empty form)', () => {
    const result = validateMacros(0, 0, 0, 0);
    expect(result.isValid).toBe(true);
    expect(result.expected).toBe(0);
  });

  it('returns valid for partial entry within tolerance', () => {
    // Only protein filled: 30P*4 = 120 expected, entered 130
    // tolerance = max(120*0.1, 20) = 20; |130-120| = 10 <= 20
    const result = validateMacros(130, 30, 0, 0);
    expect(result.isValid).toBe(true);
    expect(result.expected).toBe(120);
  });

  it('uses 20kcal minimum tolerance for small values', () => {
    // 5P*4 + 5C*4 + 2F*9 = 20 + 20 + 18 = 58
    // tolerance = max(58*0.1, 20) = 20; |75-58| = 17 <= 20
    const result = validateMacros(75, 5, 5, 2);
    expect(result.isValid).toBe(true);
    expect(result.expected).toBe(58);
  });

  it('flags when just outside 10% tolerance for larger values', () => {
    // 50P*4 + 75C*4 + 30F*9 = 200 + 300 + 270 = 770
    // tolerance = max(770*0.1, 20) = 77; |1000-770| = 230 > 77
    const result = validateMacros(1000, 50, 75, 30);
    expect(result.isValid).toBe(false);
    expect(result.expected).toBe(770);
  });
});
