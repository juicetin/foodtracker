/**
 * Jest mock for llama.rn native module.
 *
 * Prevents native module errors in test environment and provides
 * a mock context with completion and release methods.
 */

export interface LlamaContext {
  initMultimodal: jest.Mock;
  completion: jest.Mock;
  release: jest.Mock;
}

function createMockContext(): LlamaContext {
  return {
    initMultimodal: jest.fn().mockResolvedValue(undefined),
    completion: jest.fn().mockResolvedValue({ text: '{"dishes":[]}' }),
    release: jest.fn().mockResolvedValue(undefined),
  };
}

export const initLlama = jest.fn().mockResolvedValue(createMockContext());
