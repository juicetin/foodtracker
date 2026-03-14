/**
 * Tests for VLM prompt builder.
 *
 * Verifies that buildFoodPrompt generates correct prompts for
 * food identification, with and without user text context.
 */

import { buildFoodPrompt } from '../vlmPrompts';

describe('buildFoodPrompt', () => {
  it('returns string containing "Identify all food items"', () => {
    const prompt = buildFoodPrompt();
    expect(prompt).toContain('Identify all food items');
  });

  it('returns string containing "name:" and "cuisine:" and "ingredients:"', () => {
    const prompt = buildFoodPrompt();
    expect(prompt).toContain('name:');
    expect(prompt).toContain('cuisine:');
    expect(prompt).toContain('ingredients:');
  });

  it('includes user text when provided', () => {
    const prompt = buildFoodPrompt('massaman');
    expect(prompt).toContain('massaman');
  });

  it('includes "user describes" phrasing when user text provided', () => {
    const prompt = buildFoodPrompt('massaman curry');
    expect(prompt).toContain('user describes');
  });

  it('does NOT include "user describes" when no user text provided', () => {
    const prompt = buildFoodPrompt();
    expect(prompt).not.toContain('user describes');
  });
});
