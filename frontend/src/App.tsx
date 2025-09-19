import React, { Suspense, useState, useCallback } from "react";
import SplashIntro from "./components/SplashIntro";

// (optional) lazy-load your main app/dashboard
const Dashboard = React.lazy(() => import("./components/dashboard"));

export default function App() {
  const [hideSplash, setHideSplash] = useState(false); // always show on load
  const handleDone = useCallback(() => setHideSplash(true), []);

  return (
    <>
      {!hideSplash && <SplashIntro onDone={handleDone} duration={2400} perWord={350} />}

      <Suspense fallback={<div className="p-6">Loading…</div>}>
        {/* Cross-reveal: content is there; splash slides away from on top */}
        <div className={`${hideSplash ? "opacity-100" : "opacity-100"} min-h-screen bg-cover bg-center`} style={{ backgroundImage: `url(${"/bg.jpg"})` }}>
          <Dashboard />
        </div>
      </Suspense>
    </>
  );
}





