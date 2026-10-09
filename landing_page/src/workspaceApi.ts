export type DocumentPage = { page_number: number; text: string };
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

export function askDocument(docId: string, query: string): Promise<GroundedAnswer> {
  return postJson("/api/ask", { doc_id: docId, query });
}

export function narrateFrom(
  docId: string,
  page: number,
  offset: number,
): Promise<NarrationSegment> {
  return postJson("/api/narrate", { doc_id: docId, page, offset });
}

export async function transcribeRecording(blob: Blob, extension: string): Promise<string> {
  const result = await responseJson<{ transcript: string }>(
    await fetch(`/api/transcribe?extension=${extension}`, {
      method: "POST",
      headers: { "Content-Type": blob.type },
      body: blob,
    }),
  );
  return result.transcript;
}

export async function getAnswerAudio(answer: string): Promise<string> {
  const result = await postJson<{ audio_url: string }>("/api/answer-audio", { answer });
  return result.audio_url;
}
