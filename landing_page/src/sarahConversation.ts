/** Small, deterministic speech commands for the hands-free Sarah session. */

/** Accept a wake word at the start, avoiding matches inside ordinary speech. */
export function wakeQuestion(transcript: string): string | null {
  const match = transcript.trim().match(/^(?:(?:hey|hi|hello|ok|okay)[\s,.:!?-]+)?(?:sarah?|sera|saira|sairah|sayra|sehra)\b[\s,.:!?-]*/i);
  return match ? transcript.trim().slice(match[0].length).trim() : null;
}

/** Distinguish a request to continue reading from a follow-up question. */
export function wantsNarration(transcript: string): boolean {
  return /^(?:yes(?: please)?|yeah|yep|sure|continue|resume|keep reading|continue reading|continue narration|resume narration|go back to (?:the )?(?:book|narration))\b[\s.!?]*$/i
    .test(transcript.trim());
}
