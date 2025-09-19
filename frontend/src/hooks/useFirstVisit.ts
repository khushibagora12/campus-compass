import { useEffect, useState } from "react";

export function useFirstVisit(key = "seenSplashThisSession") {
  const [isFirstVisit, setIsFirstVisit] = useState<boolean>(() => {
    try { return !sessionStorage.getItem(key); } catch { return true; }
  });

  useEffect(() => {
    if (isFirstVisit) {
      try { sessionStorage.setItem(key, "true"); } catch {}
    }
  }, [isFirstVisit, key]);

  return [isFirstVisit, setIsFirstVisit] as const;
}
