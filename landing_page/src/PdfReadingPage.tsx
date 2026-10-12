import { memo, useEffect, useMemo, useRef, useState } from "react";
import { sentencesWithOffsets } from "./readerText";
import { loadPdfPageWords, pdfPageImageUrl, type PdfPageWords, type PdfWord } from "./workspaceApi";

type Props = {
  docId: string;
  pageNumber: number;
  text: string;
  activeOffset: number | null;
  playing: boolean;
  onSeek: (page: number, offset: number) => void;
};

function PdfReadingPage({
  docId, pageNumber, text, activeOffset, playing, onSeek,
}: Props) {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);
  const [geometry, setGeometry] = useState<PdfPageWords | null>(null);
  const activeWordRef = useRef<HTMLSpanElement | null>(null);
  const sentences = useMemo(() => sentencesWithOffsets(text), [text]);

  useEffect(() => {
    if (!imageLoaded) return;
    const controller = new AbortController();
    void loadPdfPageWords(docId, pageNumber, controller.signal)
      .then(setGeometry)
      .catch((error: unknown) => {
        if (!controller.signal.aborted) console.warn("PDF text overlay unavailable:", error);
      });
    return () => controller.abort();
  }, [docId, pageNumber, imageLoaded]);

  const activeSentenceIndex = activeOffset === null ? -1
    : sentences.reduce((selected, sentence, index) => sentence.start <= activeOffset ? index : selected, -1);
  const activeSentenceStart = activeSentenceIndex < 0 ? -1 : sentences[activeSentenceIndex].start;
  const activeSentenceEnd = activeSentenceIndex < 0 ? -1
    : sentences[activeSentenceIndex + 1]?.start ?? text.length;
  const activeWord = activeOffset === null ? undefined
    : geometry?.words.find((word) => word.start <= activeOffset && activeOffset < word.end)
      ?? geometry?.words.filter((word) => word.start <= activeOffset).at(-1);

  useEffect(() => {
    if (!playing || !activeWordRef.current) return;
    const word = activeWordRef.current;
    const viewport = word.closest(".pdf-book");
    if (!viewport) return;
    const wordBounds = word.getBoundingClientRect();
    const viewBounds = viewport.getBoundingClientRect();
    if (wordBounds.top < viewBounds.top + 30 || wordBounds.bottom > viewBounds.bottom - 30) {
      word.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }, [activeWord?.start, playing]);

  const seekFromWord = (word: PdfWord): void => {
    const sentence = sentences.filter((item) => item.start <= word.start).at(-1);
    onSeek(pageNumber, sentence?.start ?? word.start);
  };

  return (
    <div className="pdf-page-wrap" data-pdf-page={pageNumber}>
      <div className="pdf-page-sheet">
        <img
          className="pdf-page-image"
          src={pdfPageImageUrl(docId, pageNumber)}
          width="612" height="792" loading="lazy"
          alt={`Original PDF page ${pageNumber}`}
          onLoad={() => setImageLoaded(true)}
          onError={() => setImageError(true)}
        />
        {imageError && <p className="pdf-page-error">Page {pageNumber} could not be displayed.</p>}
        {geometry && (
          <div className="pdf-word-overlay" aria-label={`Click a sentence on page ${pageNumber} to read from it`}>
            {geometry.words.map((word) => {
              const isSentence = word.start >= activeSentenceStart && word.start < activeSentenceEnd;
              const isWord = activeWord?.start === word.start && activeWord?.end === word.end;
              return <span
                key={`${word.start}-${word.x}-${word.y}`}
                ref={isWord ? activeWordRef : undefined}
                className={`pdf-word-box${isSentence ? " current-sentence" : ""}${isWord ? " current-word" : ""}`}
                style={{
                  left: `${word.x / geometry.width * 100}%`,
                  top: `${word.y / geometry.height * 100}%`,
                  width: `${word.width / geometry.width * 100}%`,
                  height: `${word.height / geometry.height * 100}%`,
                }}
                title="Read from this sentence"
                onClick={() => seekFromWord(word)}
              />;
            })}
          </div>
        )}
      </div>
      <span className="pdf-page-number">{pageNumber}</span>
    </div>
  );
}

export default memo(PdfReadingPage);
