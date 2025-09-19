import { useEffect, useMemo, useRef, useState } from "react";

type Props = { onDone: () => void; duration?: number; perWord?: number };

export default function SplashIntro({ onDone, duration = 1200, perWord = 350 }: Props) {
  const greetings = useMemo(
    () => ["Hello","नमस्ते","হ্যালো","హలో","வணக்கம்","नमस्कार","ನಮಸ್ಕಾರ"],
    []
  );
  const [index, setIndex] = useState(0);
  const [slideUp, setSlideUp] = useState(false);
  const cycleRef = useRef<number | null>(null);

  useEffect(() => {
    cycleRef.current = window.setInterval(
      () => setIndex(i => (i + 1) % greetings.length),
      perWord
    );
    return () => { if (cycleRef.current) clearInterval(cycleRef.current); };
  }, [greetings.length, perWord]);

  useEffect(() => {
    const t = window.setTimeout(() => {
      setSlideUp(true);
      window.setTimeout(onDone, 600);
    }, duration);
    return () => clearTimeout(t);
  }, [duration, onDone]);

  return (
    <div
      aria-hidden
      className={[
        "fixed inset-0 z-[9999] flex items-center justify-center",
        "bg-gradient-to-br from-[#4e1eeb] to-[#e0dced]",
        "transition-transform duration-700 ease-in-out",
        slideUp ? "-translate-y-full" : "translate-y-0"
      ].join(" ")}
    >
      <div className="text-white text-5xl md:text-7xl font-semibold tracking-wide">
        <WordSwitcher words={greetings} index={index} />
      </div>
    </div>
  );
}

function WordSwitcher({ words, index }: { words: string[]; index: number }) {
  return (
    <div className="relative h-[1.2em] w-[10ch] flex items-center justify-center">
      {words.map((w, i) => (
        <span
          key={w + i}
          className={[
            "absolute transition-opacity duration-200",
            i === index ? "opacity-100" : "opacity-0"
          ].join(" ")}
        >
          {w}
        </span>
      ))}
    </div>
  );
}
