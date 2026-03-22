/**
 * Pure-JS WordPiece tokenizer for BERT-family models.
 *
 * Produces token IDs, attention masks, and padding compatible with
 * MiniLM / all-MiniLM-L6-v2 TFLite models.
 */

const PUNCT_RE = /([.,!?;:'"()\[\]{}\/\-])/g;

/**
 * Tokenize text using WordPiece vocabulary.
 *
 * @param text - Input text (e.g. "chicken breast")
 * @param vocab - Map of token string -> token ID
 * @param maxLen - Maximum sequence length (padding target)
 * @returns inputIds and attentionMask as Int32Arrays of length maxLen
 */
export function tokenize(
  text: string,
  vocab: Map<string, number>,
  maxLen: number,
): { inputIds: Int32Array; attentionMask: Int32Array } {
  const CLS_ID = vocab.get('[CLS]')!;
  const SEP_ID = vocab.get('[SEP]')!;
  const UNK_ID = vocab.get('[UNK]')!;

  // Step 1: Lowercase
  const lower = text.toLowerCase();

  // Step 2: Basic tokenization — split punctuation, then whitespace
  const words = lower
    .replace(PUNCT_RE, ' $1 ')
    .split(/\s+/)
    .filter((w) => w.length > 0);

  // Step 3: WordPiece tokenization
  const tokens: number[] = [];
  const maxTokens = maxLen - 2; // reserve [CLS] and [SEP]

  for (const word of words) {
    if (tokens.length >= maxTokens) break;

    let start = 0;
    while (start < word.length) {
      if (tokens.length >= maxTokens) break;

      let matched = false;
      for (let end = word.length; end > start; end--) {
        const substr = word.slice(start, end);
        const piece = start > 0 ? `##${substr}` : substr;

        if (vocab.has(piece)) {
          tokens.push(vocab.get(piece)!);
          start = end;
          matched = true;
          break;
        }
      }

      if (!matched) {
        // No subword match — emit [UNK] and advance by 1 character
        tokens.push(UNK_ID);
        start += 1;
      }
    }
  }

  // Step 4: Build output arrays
  const seqLen = Math.min(tokens.length + 2, maxLen); // +2 for [CLS] and [SEP]

  const inputIds = new Int32Array(maxLen); // zero-filled (PAD=0)
  const attentionMask = new Int32Array(maxLen);

  inputIds[0] = CLS_ID;
  attentionMask[0] = 1;

  for (let i = 0; i < seqLen - 2; i++) {
    inputIds[i + 1] = tokens[i];
    attentionMask[i + 1] = 1;
  }

  inputIds[seqLen - 1] = SEP_ID;
  attentionMask[seqLen - 1] = 1;

  return { inputIds, attentionMask };
}
