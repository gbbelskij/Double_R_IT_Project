import { useSyncExternalStore } from "react";

const subscribe = (callback: () => void) => {
  window.addEventListener("resize", callback);

  return () => window.removeEventListener("resize", callback);
};

const getSnapshot = () => window.innerWidth;

export function useWindowSize() {
  const width = useSyncExternalStore(subscribe, getSnapshot);

  const isSmallMobile = width <= 375;
  const isMobile = width <= 768;
  const isTablet = width > 768 && width <= 1024;
  const isDesktop = width > 1024;

  return {
    width,
    isSmallMobile,
    isMobile,
    isTablet,
    isDesktop,
  };
}
