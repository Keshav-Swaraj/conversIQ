export type Sentence = { start: number; text: string };

/** Split visible PDF text while retaining exact offsets for narration seeking. */
export function sentencesWithOffsets(text: string): Sentence[] {
  const result: Sentence[] = [];
  const endings = /[.!?]["'”’)]*(?=\s|$)/g;
  let start = 0;
  for (const match of text.matchAll(endings)) {
    const end = match.index + match[0].length;
    const part = text.slice(start, end);
    if (part.trim()) result.push({ start: start + part.search(/\S/), text: part });
    start = end;
  }
  const remainder = text.slice(start);
  if (remainder.trim()) {
    result.push({ start: start + remainder.search(/\S/), text: remainder });
  }
  return result;
}
