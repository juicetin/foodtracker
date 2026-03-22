import { tokenize } from '../wordpieceTokenizer';

/**
 * Minimal test vocab for WordPiece tokenizer tests.
 * Uses realistic BERT-style special token IDs.
 */
function buildTestVocab(): Map<string, number> {
  return new Map<string, number>([
    ['[PAD]', 0],
    ['[UNK]', 100],
    ['[CLS]', 101],
    ['[SEP]', 102],
    ['chicken', 2000],
    ['breast', 2001],
    ['pad', 2002],
    ['thai', 2003],
    ['ton', 2004],
    ['##kat', 2005],
    ['##su', 2006],
    [',', 2007],
    ['grilled', 2008],
    ['a', 2009],
    ['b', 2010],
    ['c', 2011],
    ['d', 2012],
    ['e', 2013],
    ['f', 2014],
    ['g', 2015],
    ['h', 2016],
  ]);
}

describe('wordpieceTokenizer', () => {
  const vocab = buildTestVocab();
  const MAX_LEN = 128;

  it('produces [CLS] at position 0 and [SEP] after last real token', () => {
    const { inputIds } = tokenize('chicken', vocab, MAX_LEN);
    expect(inputIds[0]).toBe(101); // [CLS]
    // Find [SEP] — should be after "chicken" token
    expect(inputIds[1]).toBe(2000); // chicken
    expect(inputIds[2]).toBe(102); // [SEP]
  });

  it('produces attentionMask with 1s for real tokens and 0s for padding', () => {
    const { inputIds, attentionMask } = tokenize('chicken', vocab, MAX_LEN);
    // [CLS] chicken [SEP] = 3 real tokens
    expect(attentionMask[0]).toBe(1);
    expect(attentionMask[1]).toBe(1);
    expect(attentionMask[2]).toBe(1);
    expect(attentionMask[3]).toBe(0); // padding starts
    expect(attentionMask[MAX_LEN - 1]).toBe(0);

    // inputIds padding should be 0
    expect(inputIds[3]).toBe(0);
  });

  it('lowercases input (case insensitive)', () => {
    const lower = tokenize('chicken', vocab, MAX_LEN);
    const upper = tokenize('CHICKEN', vocab, MAX_LEN);
    expect(Array.from(lower.inputIds)).toEqual(Array.from(upper.inputIds));
  });

  it('handles multi-word input (splits on whitespace)', () => {
    const { inputIds } = tokenize('chicken breast', vocab, MAX_LEN);
    expect(inputIds[0]).toBe(101); // [CLS]
    expect(inputIds[1]).toBe(2000); // chicken
    expect(inputIds[2]).toBe(2001); // breast
    expect(inputIds[3]).toBe(102); // [SEP]
  });

  it('splits punctuation into separate tokens', () => {
    const { inputIds } = tokenize('chicken, grilled', vocab, MAX_LEN);
    expect(inputIds[0]).toBe(101); // [CLS]
    expect(inputIds[1]).toBe(2000); // chicken
    expect(inputIds[2]).toBe(2007); // ,
    expect(inputIds[3]).toBe(2008); // grilled
    expect(inputIds[4]).toBe(102); // [SEP]
  });

  it('performs WordPiece subword splitting', () => {
    const { inputIds } = tokenize('tonkatsu', vocab, MAX_LEN);
    expect(inputIds[0]).toBe(101); // [CLS]
    expect(inputIds[1]).toBe(2004); // ton
    expect(inputIds[2]).toBe(2005); // ##kat
    expect(inputIds[3]).toBe(2006); // ##su
    expect(inputIds[4]).toBe(102); // [SEP]
  });

  it('uses [UNK] for truly unknown characters', () => {
    const { inputIds } = tokenize('xyz', vocab, MAX_LEN);
    expect(inputIds[0]).toBe(101); // [CLS]
    // "xyz" is unknown — no subword matches, each char gets [UNK]
    // At least one UNK should appear
    const ids = Array.from(inputIds).slice(1);
    const sepIdx = ids.indexOf(102);
    const contentIds = ids.slice(0, sepIdx);
    expect(contentIds.some((id) => id === 100)).toBe(true); // [UNK]
  });

  it('truncates at maxLen', () => {
    // Use maxLen=6: [CLS] + 4 tokens max + [SEP]
    const { inputIds, attentionMask } = tokenize(
      'a b c d e f g h',
      vocab,
      6,
    );
    expect(inputIds.length).toBe(6);
    expect(attentionMask.length).toBe(6);
    // First should be [CLS], last real should be [SEP]
    expect(inputIds[0]).toBe(101);
    // [SEP] should be at position 5 (maxLen-1) or earlier
    const ids = Array.from(inputIds);
    expect(ids.includes(102)).toBe(true);
  });

  it('returns Int32Array of length maxLen for both outputs', () => {
    const { inputIds, attentionMask } = tokenize('chicken', vocab, MAX_LEN);
    expect(inputIds).toBeInstanceOf(Int32Array);
    expect(attentionMask).toBeInstanceOf(Int32Array);
    expect(inputIds.length).toBe(MAX_LEN);
    expect(attentionMask.length).toBe(MAX_LEN);
  });
});
