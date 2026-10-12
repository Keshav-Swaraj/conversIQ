export type DocumentPage = { page_number: number; text: string };
export type PdfWord = {
  start: number; end: number; x: number; y: number; width: number; height: number;
};
export type PdfPageWords = { width: number; height: number; words: PdfWord[] };
export type DocumentSection = {
  id: string;
  title: string;
  page: number;
  end_page: number;
};
export type WorkspaceDocument = {
  doc_id: string;
  filename: string;
  num_pages: number;
  sections: DocumentSection[];
  pages: DocumentPage[];
};
export type GroundedAnswer = { answer: string; relevant_pages: number[] };
export type ConversationTurn = { question: string; answer: string };
export type AnswerEvent =
  | { type: "sources"; relevant_pages: number[] }
  | { type: "delta"; text: string }
  | { type: "audio"; pcm_base64: string }
  | { type: "done" }
  | { type: "error"; message: string };
export type NarrationSegment =
  | { complete: true }
  | {
      complete: false;
      page: number;
      offset: number;
      text: string;
      next_page: number;
      next_offset: number;
      next_complete: boolean;
      audio_url: string;
    };

async function responseJson<T>(response: Response): Promise<T> {
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status}).`);
  return body as T;
}

async function postJson<T>(path: string, payload: object): Promise<T> {
  return responseJson<T>(
    await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  );
}

export async function uploadPdf(file: File): Promise<WorkspaceDocument> {
  if (!file.name.toLowerCase().endsWith(".pdf")) throw new Error("Choose a PDF file.");
  return responseJson<WorkspaceDocument>(
    await fetch(`/api/upload?filename=${encodeURIComponent(file.name)}`, {
      method: "POST",
      headers: { "Content-Type": "application/pdf" },
      body: file,
    }),
  );
}

export async function loadDocument(id: string): Promise<WorkspaceDocument> {
  return responseJson<WorkspaceDocument>(
    await fetch(`/api/document?id=${encodeURIComponent(id)}`),
  );
}

export function pdfPageImageUrl(docId: string, pageNumber: number): string {
  return `/api/pdf-page?id=${encodeURIComponent(docId)}&page=${pageNumber}`;
}

export async function loadPdfPageWords(
  docId: string, pageNumber: number, signal?: AbortSignal,
): Promise<PdfPageWords> {
  return responseJson<PdfPageWords>(await fetch(
    `/api/pdf-words?id=${encodeURIComponent(docId)}&page=${pageNumber}`,
    { signal },
  ));
}

export function askDocument(docId: string, query: string): Promise<GroundedAnswer> {
  return postJson("/api/ask", { doc_id: docId, query });
}

async function* streamAnswerEvents(
  path: string, payload: object, signal: AbortSignal,
): AsyncGenerator<AnswerEvent> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok) await responseJson<never>(response);
  if (!response.body) throw new Error("This browser cannot stream the answer.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line) as AnswerEvent;
        if (event.type === "error") throw new Error(event.message);
        yield event;
      }
      if (done) {
        if (buffer.trim()) throw new Error("Answer stream ended with incomplete data.");
        return;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export function streamLiveAnswer(
  docId: string, query: string, signal: AbortSignal, history: ConversationTurn[] = [],
): AsyncGenerator<AnswerEvent> {
  return streamAnswerEvents("/api/ask-live-stream", { doc_id: docId, query, history }, signal);
}

export function streamDocumentAnswer(
  docId: string, query: string, signal: AbortSignal,
): AsyncGenerator<AnswerEvent> {
  return streamAnswerEvents("/api/ask-stream", { doc_id: docId, query }, signal);
}

export function streamSarahSpeech(text: string, signal: AbortSignal): AsyncGenerator<AnswerEvent> {
  return streamAnswerEvents("/api/sarah-speech", { text }, signal);
}

export function narrateFrom(
  docId: string,
  page: number,
  offset: number,
): Promise<NarrationSegment> {
  return postJson("/api/narrate", { doc_id: docId, page, offset });
}

export async function transcribeRecording(blob: Blob, extension: string, wakeWord = false): Promise<string> {
  const result = await responseJson<{ transcript: string }>(
    await fetch(`/api/transcribe?extension=${extension}${wakeWord ? "&wake=1" : ""}`, {
      method: "POST",
      headers: { "Content-Type": blob.type },
      body: blob,
    }),
  );
  return result.transcript;
}

export async function getAnswerAudio(answer: string, signal?: AbortSignal): Promise<string> {
  const result = await responseJson<{ audio_url: string }>(await fetch("/api/answer-audio", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ answer }),
    signal,
  }));
  return result.audio_url;
}
