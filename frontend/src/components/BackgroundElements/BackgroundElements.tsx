import React, { useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import classNames from "classnames";

import { useWindowSize } from "@hooks/useWindowSize";

import Blob from "./components/Blob/Blob";

import { BackgroundElementsProps } from "./BackgroundElements.props";

import classes from "./BackgroundElements.module.css";

const defaultColors = ["var(--blue-ribbon-100)", "var(--flush-orange-100)"];

const BackgroundElements = ({
  targetRef,
  color = "",
  count = 2,
  blobsExtraPoints,
  blobsRandomness,
  blobsSize,
  styles,
}: BackgroundElementsProps): React.ReactPortal | null => {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [style, setStyle] = useState<React.CSSProperties>({});

  const { isSmallMobile } = useWindowSize();

  useLayoutEffect(() => {
    const elem = targetRef?.current;
    const wrapper = wrapperRef.current;

    if (!elem || !wrapper) {
      return;
    }

    const updatePosition = () => {
      const rect = elem.getBoundingClientRect();

      setStyle({
        width: `${rect.width}px`,
        height: `${rect.height}px`,
        top: `${rect.top + window.scrollY}px`,
        left: `${rect.left + window.scrollX}px`,
      });
    };

    updatePosition();

    const resizeObserver = new ResizeObserver(updatePosition);
    resizeObserver.observe(elem);

    window.addEventListener("scroll", updatePosition, true);
    window.addEventListener("resize", updatePosition);

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener("scroll", updatePosition, true);
      window.removeEventListener("resize", updatePosition);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [targetRef, targetRef?.current]);

  const root = document.getElementById("background-root");

  if (!root) {
    return null;
  }

  return createPortal(
    <div
      ref={wrapperRef}
      style={style}
      className={classNames(classes.BackgroundElements, {
        [styles!]: styles,
      })}
    >
      {Array.from({ length: count }).map((_, i) => (
        <Blob
          key={i}
          color={color || defaultColors[i % defaultColors.length]}
          extraPoints={blobsExtraPoints}
          randomness={blobsRandomness}
          size={blobsSize || (isSmallMobile ? 200 : 300)}
        />
      ))}
    </div>,
    root
  );
};

export default BackgroundElements;
